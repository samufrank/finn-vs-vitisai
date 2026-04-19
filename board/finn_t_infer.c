/*
 * finn_t_infer.c — board-side hot-path for the FINN-T transformer.
 *
 * Compiled as a shared library and loaded via ctypes from benchmark.py.
 * Python keeps ownership of: bitstream load (FINNDMAOverlay), CMA buffer
 * allocation (pynq.allocate), MMIO mapping for the AXI-DMA register blocks,
 * data loading, and JSON results. The C side does just the per-inference
 * sequence (signal -> ibuf, DMA trigger, poll, INT5 decode, transpose +
 * GAP + MatMul + argmax) so the FPGA's measured throughput isn't drowned
 * out by Python overhead.
 *
 * Two API tiers:
 *
 *   v1 (finn_t_infer_one / finn_t_infer_batch)
 *       Cached buffers, single-buffered, explicit dc cvac/civac per sample.
 *       Always available. Used as fallback and as the regression baseline.
 *
 *   v2 (finn_t_runner_init + finn_t_infer_batch_v2 + finn_t_runner_destroy)
 *       Configurable at init time:
 *         use_cache_ops: 0 = caller's buffers are uncached (skip dc cvac/
 *                            civac entirely). PYNQ buffers allocated with
 *                            cacheable=False land here.
 *                        1 = caller's buffers are cached, do dc cvac/civac.
 *         double_buffer: 0 = single ibuf/obuf, lockstep DMA-then-tail.
 *                        1 = two ibuf/obuf pairs alternating; CPU stages
 *                            sample N+1 and tails sample N-1 *while* the
 *                            FPGA is processing sample N.
 *       The performance-relevant combination is (use_cache_ops=0,
 *       double_buffer=1) — both optimizations on. The other combinations
 *       exist so the user can disable one at a time and measure the
 *       contribution of each.
 *
 * Build (on board):
 *   gcc -O2 -shared -fPIC -Wall -o libfinn_t_infer.so finn_t_infer.c
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define ACCEL_INPUT_BYTES  (1 * 1 * 1024 * 2 * 4)   /* float32 */
#define ACCEL_OUT_SEQ      64
#define ACCEL_OUT_CHAN     96
#define ACCEL_OUT_BYTES    (ACCEL_OUT_SEQ * ACCEL_OUT_CHAN)  /* uint8 INT5 */
#define NUM_CLASSES        24

#define DMA_REG_ADDR_LO    (0x10 / 4)   /* 32-bit register file index */
#define DMA_REG_ADDR_HI    (0x14 / 4)
#define DMA_REG_COUNT      (0x1C / 4)
#define DMA_REG_CTRL       (0x00 / 4)
#define DMA_AP_DONE_BIT    0x2

#define CACHE_LINE_BYTES   64

/* ---- ARMv8 cache maintenance (unprivileged on Linux ARM64; UCI=1) ---- */

static inline void dcache_clean(const void *addr, size_t size)
{
    uintptr_t start = (uintptr_t)addr & ~((uintptr_t)CACHE_LINE_BYTES - 1);
    uintptr_t end = (uintptr_t)addr + size;
    for (uintptr_t a = start; a < end; a += CACHE_LINE_BYTES) {
        __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
}

static inline void dcache_invalidate(const void *addr, size_t size)
{
    /* DC CIVAC = clean and invalidate; unprivileged. obuf is never CPU-
     * dirty (the CPU only reads it), so the clean is a no-op and we get
     * the invalidate. DC IVAC alone is privileged on most kernels. */
    uintptr_t start = (uintptr_t)addr & ~((uintptr_t)CACHE_LINE_BYTES - 1);
    uintptr_t end = (uintptr_t)addr + size;
    for (uintptr_t a = start; a < end; a += CACHE_LINE_BYTES) {
        __asm__ volatile("dc civac, %0" :: "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
}

/* ---- DMA register helpers ---- */

static inline void mmio_write(volatile uint32_t *base, unsigned idx, uint32_t v)
{
    base[idx] = v;
}

static inline uint32_t mmio_read(volatile uint32_t *base, unsigned idx)
{
    return base[idx];
}

/* ---- CPU tail (shared by v1 and v2) ----
 * Layout of obuf is [seq=64, chan=96] uint8 storing INT5 (raw 5-bit
 * unsigned bin index packed into uint8). Sign-extend via the
 *   value > 15 ? value - 32 : value
 * rule, then transpose + GAP collapse to gap[96] = mean over seq dim,
 * matmul against W_cls [96,24] int8, argmax over 24 classes. */
static int cpu_tail(const uint8_t *raw, const int8_t *W_cls)
{
    float gap[ACCEL_OUT_CHAN];
    for (int c = 0; c < ACCEL_OUT_CHAN; c++) {
        int32_t acc = 0;
        for (int s = 0; s < ACCEL_OUT_SEQ; s++) {
            int v = (int)raw[s * ACCEL_OUT_CHAN + c];
            if (v > 15) v -= 32;
            acc += v;
        }
        gap[c] = (float)acc / (float)ACCEL_OUT_SEQ;
    }
    int best_idx = 0;
    float best_score = -1e30f;
    for (int o = 0; o < NUM_CLASSES; o++) {
        float score = 0.0f;
        for (int i = 0; i < ACCEL_OUT_CHAN; i++) {
            score += gap[i] * (float)W_cls[i * NUM_CLASSES + o];
        }
        if (score > best_score) {
            best_score = score;
            best_idx = o;
        }
    }
    return best_idx;
}

/* ============================================================
 * v1 API — cached, single-buffered. Preserved unchanged.
 * ============================================================ */

int finn_t_infer_one(
    const void *signal,
    void *ibuf_virt,
    uint64_t ibuf_phys,
    void *obuf_virt,
    uint64_t obuf_phys,
    void *idma_mmio,
    void *odma_mmio,
    const int8_t *W_cls)
{
    volatile uint32_t *idma = (volatile uint32_t *)idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)odma_mmio;

    memcpy(ibuf_virt, signal, ACCEL_INPUT_BYTES);
    dcache_clean(ibuf_virt, ACCEL_INPUT_BYTES);

    mmio_write(odma, DMA_REG_ADDR_LO, (uint32_t)(obuf_phys & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_ADDR_HI, (uint32_t)((obuf_phys >> 32) & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_COUNT,   1);
    mmio_write(odma, DMA_REG_CTRL,    1);

    mmio_write(idma, DMA_REG_ADDR_LO, (uint32_t)(ibuf_phys & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_ADDR_HI, (uint32_t)((ibuf_phys >> 32) & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_COUNT,   1);
    mmio_write(idma, DMA_REG_CTRL,    1);

    while ((mmio_read(odma, DMA_REG_CTRL) & DMA_AP_DONE_BIT) == 0) { }

    dcache_invalidate(obuf_virt, ACCEL_OUT_BYTES);
    return cpu_tail((const uint8_t *)obuf_virt, W_cls);
}

int finn_t_infer_batch(
    const void *signals,
    const int32_t *labels,
    int n_samples,
    int32_t *predictions_out,
    void *ibuf_virt,
    uint64_t ibuf_phys,
    void *obuf_virt,
    uint64_t obuf_phys,
    void *idma_mmio,
    void *odma_mmio,
    const int8_t *W_cls)
{
    const uint8_t *base = (const uint8_t *)signals;
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        int pred = finn_t_infer_one(
            base + (size_t)i * ACCEL_INPUT_BYTES,
            ibuf_virt, ibuf_phys, obuf_virt, obuf_phys,
            idma_mmio, odma_mmio, W_cls);
        predictions_out[i] = (int32_t)pred;
        if (labels && pred == labels[i]) correct++;
    }
    return correct;
}

/* ============================================================
 * v2 API — configurable cached/uncached, single/double buffer.
 * ============================================================
 *
 * Pattern:
 *   1. Caller (Python) allocates the buffers and any MMIO mappings.
 *   2. finn_t_runner_init records pointers + flags into a static state.
 *   3. finn_t_infer_batch_v2 runs the hot loop.
 *   4. finn_t_runner_destroy clears the state.
 *
 * Single static instance: only one v2 runner can be active at a time.
 * That matches the one-bitstream-per-process reality — no need for a
 * handle-based API. */

typedef struct {
    void     *ibuf_virt[2];
    uint64_t  ibuf_phys[2];
    void     *obuf_virt[2];
    uint64_t  obuf_phys[2];
    void     *idma_mmio;
    void     *odma_mmio;
    int       n_buffers;       /* 1 = single-buffered, 2 = double */
    int       use_cache_ops;   /* 0 = caller's buffers are uncached */
    int       initialized;
} runner_state_t;

static runner_state_t g_runner = {0};

int finn_t_runner_init(
    int       n_buffers,        /* 1 or 2 */
    int       use_cache_ops,    /* 0 = uncached buffers, 1 = cached */
    void     *ibuf_a_virt,
    uint64_t  ibuf_a_phys,
    void     *ibuf_b_virt,      /* may be NULL if n_buffers == 1 */
    uint64_t  ibuf_b_phys,
    void     *obuf_a_virt,
    uint64_t  obuf_a_phys,
    void     *obuf_b_virt,      /* may be NULL if n_buffers == 1 */
    uint64_t  obuf_b_phys,
    void     *idma_mmio,
    void     *odma_mmio)
{
    if (n_buffers != 1 && n_buffers != 2) return -1;
    if (n_buffers == 2 && (ibuf_b_virt == NULL || obuf_b_virt == NULL)) return -2;
    if (ibuf_a_virt == NULL || obuf_a_virt == NULL) return -3;
    if (idma_mmio == NULL || odma_mmio == NULL) return -4;

    g_runner.n_buffers     = n_buffers;
    g_runner.use_cache_ops = use_cache_ops;
    g_runner.ibuf_virt[0]  = ibuf_a_virt;
    g_runner.ibuf_phys[0]  = ibuf_a_phys;
    g_runner.obuf_virt[0]  = obuf_a_virt;
    g_runner.obuf_phys[0]  = obuf_a_phys;
    g_runner.ibuf_virt[1]  = (n_buffers == 2) ? ibuf_b_virt : ibuf_a_virt;
    g_runner.ibuf_phys[1]  = (n_buffers == 2) ? ibuf_b_phys : ibuf_a_phys;
    g_runner.obuf_virt[1]  = (n_buffers == 2) ? obuf_b_virt : obuf_a_virt;
    g_runner.obuf_phys[1]  = (n_buffers == 2) ? obuf_b_phys : obuf_a_phys;
    g_runner.idma_mmio     = idma_mmio;
    g_runner.odma_mmio     = odma_mmio;
    g_runner.initialized   = 1;
    return 0;
}

int finn_t_runner_destroy(void)
{
    memset(&g_runner, 0, sizeof(g_runner));
    return 0;
}

/* Helper: program both DMAs to point at slot's buffers and AP_START. */
static inline void trigger_dma(int slot)
{
    volatile uint32_t *idma = (volatile uint32_t *)g_runner.idma_mmio;
    volatile uint32_t *odma = (volatile uint32_t *)g_runner.odma_mmio;
    uint64_t ip = g_runner.ibuf_phys[slot];
    uint64_t op = g_runner.obuf_phys[slot];

    /* Same order as v1 / driver.py: arm output DMA before input. */
    mmio_write(odma, DMA_REG_ADDR_LO, (uint32_t)(op & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_ADDR_HI, (uint32_t)((op >> 32) & 0xFFFFFFFFu));
    mmio_write(odma, DMA_REG_COUNT,   1);
    mmio_write(odma, DMA_REG_CTRL,    1);

    mmio_write(idma, DMA_REG_ADDR_LO, (uint32_t)(ip & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_ADDR_HI, (uint32_t)((ip >> 32) & 0xFFFFFFFFu));
    mmio_write(idma, DMA_REG_COUNT,   1);
    mmio_write(idma, DMA_REG_CTRL,    1);
}

static inline void wait_dma(void)
{
    volatile uint32_t *odma = (volatile uint32_t *)g_runner.odma_mmio;
    while ((mmio_read(odma, DMA_REG_CTRL) & DMA_AP_DONE_BIT) == 0) { }
}

/* ---- Single-buffered v2 hot loop: one ibuf, one obuf. Cache ops are
 *      controlled by use_cache_ops. Identical structure to v1 batch but
 *      reads buffer addrs from g_runner instead of per-call args. */
static int infer_batch_single(
    const void *signals, const int32_t *labels, int n_samples,
    int32_t *predictions_out, const int8_t *W_cls)
{
    const uint8_t *src = (const uint8_t *)signals;
    void *ib = g_runner.ibuf_virt[0];
    void *ob = g_runner.obuf_virt[0];
    int correct = 0;

    for (int i = 0; i < n_samples; i++) {
        memcpy(ib, src + (size_t)i * ACCEL_INPUT_BYTES, ACCEL_INPUT_BYTES);
        if (g_runner.use_cache_ops) dcache_clean(ib, ACCEL_INPUT_BYTES);

        trigger_dma(0);
        wait_dma();

        if (g_runner.use_cache_ops) dcache_invalidate(ob, ACCEL_OUT_BYTES);
        int pred = cpu_tail((const uint8_t *)ob, W_cls);
        predictions_out[i] = (int32_t)pred;
        if (labels && pred == labels[i]) correct++;
    }
    return correct;
}

/* ---- Double-buffered v2 hot loop. Pipelining:
 *
 *   slot   = current DMA's slot
 *   nslot  = other slot, used for staging next input + reading prev output
 *
 *   prologue:
 *     stage signal[0] -> ibuf[0], trigger DMA on slot 0
 *
 *   loop i = 1..N-1:
 *     stage signal[i] -> ibuf[nslot]   (overlaps with current DMA)
 *     wait current DMA done
 *     trigger DMA on nslot              (starts the next FPGA work)
 *     run cpu tail on obuf[slot]        (overlaps with the DMA we just started)
 *     write prediction[i-1]
 *     swap slot <-> nslot
 *
 *   epilogue:
 *     wait final DMA, run cpu tail on its obuf, write prediction[N-1]
 *
 * The CPU work that overlaps with FPGA latency is one memcpy(8 KB) +
 * optional dcache_clean + cpu_tail (~5 µs). On a Cortex-A53 polling for
 * a 250-ish-µs FPGA inference, that's roughly 20-30 µs of CPU work
 * hidden under each iteration's wait.
 */
static int infer_batch_double(
    const void *signals, const int32_t *labels, int n_samples,
    int32_t *predictions_out, const int8_t *W_cls)
{
    const uint8_t *src = (const uint8_t *)signals;
    int correct = 0;

    if (n_samples == 0) return 0;

    /* Single-sample fast path: degenerate to v1-style sequential. */
    if (n_samples == 1) {
        memcpy(g_runner.ibuf_virt[0], src, ACCEL_INPUT_BYTES);
        if (g_runner.use_cache_ops)
            dcache_clean(g_runner.ibuf_virt[0], ACCEL_INPUT_BYTES);
        trigger_dma(0);
        wait_dma();
        if (g_runner.use_cache_ops)
            dcache_invalidate(g_runner.obuf_virt[0], ACCEL_OUT_BYTES);
        int pred = cpu_tail((const uint8_t *)g_runner.obuf_virt[0], W_cls);
        predictions_out[0] = (int32_t)pred;
        return (labels && pred == labels[0]) ? 1 : 0;
    }

    /* Prologue: stage sample 0, kick off the first DMA. */
    int slot = 0;
    memcpy(g_runner.ibuf_virt[slot], src, ACCEL_INPUT_BYTES);
    if (g_runner.use_cache_ops)
        dcache_clean(g_runner.ibuf_virt[slot], ACCEL_INPUT_BYTES);
    trigger_dma(slot);

    /* Steady state. At loop top, `slot` is the in-flight DMA's slot. */
    for (int i = 1; i < n_samples; i++) {
        int nslot = 1 - slot;

        /* Stage sample i in the OTHER buffer pair while the current DMA
         * is still running. */
        memcpy(g_runner.ibuf_virt[nslot],
               src + (size_t)i * ACCEL_INPUT_BYTES,
               ACCEL_INPUT_BYTES);
        if (g_runner.use_cache_ops)
            dcache_clean(g_runner.ibuf_virt[nslot], ACCEL_INPUT_BYTES);

        /* Synchronize on the current DMA's completion. */
        wait_dma();

        /* Hand off to the next DMA immediately so the FPGA stays busy. */
        trigger_dma(nslot);

        /* Run CPU tail on the just-completed sample i-1, in parallel
         * with the DMA we just kicked off. */
        if (g_runner.use_cache_ops)
            dcache_invalidate(g_runner.obuf_virt[slot], ACCEL_OUT_BYTES);
        int pred = cpu_tail((const uint8_t *)g_runner.obuf_virt[slot], W_cls);
        predictions_out[i - 1] = (int32_t)pred;
        if (labels && pred == labels[i - 1]) correct++;

        slot = nslot;
    }

    /* Epilogue: wait for the last DMA and tail it. */
    wait_dma();
    if (g_runner.use_cache_ops)
        dcache_invalidate(g_runner.obuf_virt[slot], ACCEL_OUT_BYTES);
    int pred = cpu_tail((const uint8_t *)g_runner.obuf_virt[slot], W_cls);
    predictions_out[n_samples - 1] = (int32_t)pred;
    if (labels && pred == labels[n_samples - 1]) correct++;

    return correct;
}

int finn_t_infer_batch_v2(
    const void *signals,
    const int32_t *labels,
    int n_samples,
    int32_t *predictions_out,
    const int8_t *W_cls)
{
    if (!g_runner.initialized) return -1;
    if (n_samples < 0) return -2;
    if (g_runner.n_buffers == 2) {
        return infer_batch_double(signals, labels, n_samples,
                                  predictions_out, W_cls);
    }
    return infer_batch_single(signals, labels, n_samples,
                              predictions_out, W_cls);
}
