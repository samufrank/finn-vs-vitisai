/*
 * VTA PYNQ driver — XRT backend (Vivado HLS 2020.1 register layout)
 *
 * Uses XRT's C API for DMA buffer allocation and cache management.
 * Replaces libcma/xlnk (deprecated) and dma_heap (broken cache ops).
 *
 * Register layout: Vivado HLS 2020.1 generates 32-bit pointer registers
 * with offsets matching the original VTA defines exactly:
 *   Fetch:   insn_count=0x10, insns=0x18
 *   Load:    inputs=0x10, weights=0x18
 *   Compute: done_wr=0x10, done_rd=0x18, done_vld=0x1c, uops=0x20, biases=0x28
 *   Store:   outputs=0x10
 *
 * Done polling: all four modules run with AUTORESTART except fetch (single-
 * shot). Fetch's ap_idle (bit 0x4 of control) asserts after the last insn
 * dispatches. Load/store block on empty HLS stream FIFO reads between
 * iterations, so their ap_idle NEVER asserts under AUTORESTART (observed
 * session 6). Compute exposes ap_done via HLS 2020.1 function-return-value
 * registers: done_wr @ 0x10 (ack, written to 0 before start), done_rd @ 0x18
 * (value), done_vld @ 0x1c (valid flag). After the compiled module's FINISH
 * instruction completes, compute asserts done_vld and it latches until the
 * next ack. We poll done_vld, then apply a small fixed drain (~50us) for the
 * store DMA pipeline tail. If done_vld times out, we fall back to the old
 * insn_count-scaled usleep as a safety net.
 *
 * Previous versions of this driver used only a blind usleep(insn_count * 2)
 * drain, which was insufficient for CNN workloads and caused non-deterministic
 * GEMM outputs on identical inputs (compute+store not fully drained before
 * readback). The compute-done_vld poll fixes that.
 *
 * XRT is already initialized by PYNQ's overlay loading, which configures
 * the PSDDR memory bank. After that, XRT allocations go to CMA.
 */

#include <vta/driver.h>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <time.h>
#include <map>
#include <mutex>
#include <atomic>

/* XRT C API headers */
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

#include "pynq_driver.h"

/* ---- Allocation tracking ---- */

struct XRTAlloc {
  xrtBufferHandle bo_handle;
  size_t size;
  uint64_t phys_addr;
};

static std::map<void*, XRTAlloc> alloc_map;
static std::mutex alloc_mutex;
static xrtDeviceHandle xrt_device = nullptr;

static xrtDeviceHandle get_xrt_device() {
  if (xrt_device == nullptr) {
    xrt_device = xrtDeviceOpen(0);
    if (xrt_device == nullptr) {
      fprintf(stderr, "VTA: failed to open XRT device\n");
    }
  }
  return xrt_device;
}

/*
 * Find the allocation that contains the given address.
 * Caller must hold alloc_mutex.
 */
static std::map<void*, XRTAlloc>::iterator find_containing_alloc(void* addr) {
  uintptr_t target = (uintptr_t)addr;
  for (auto it = alloc_map.begin(); it != alloc_map.end(); ++it) {
    uintptr_t base = (uintptr_t)it->first;
    if (target >= base && target < base + it->second.size) {
      return it;
    }
  }
  return alloc_map.end();
}

/* ---- VTA Memory Interface ---- */

void* VTAMemAlloc(size_t size, int cached) {
  assert(size <= VTA_MAX_XFER);
  // XRT places <64KB allocs in high memory (>4GB), breaking VTA's 32-bit phys addr.
  // Pad to 64KB to force CMA placement.
  size_t alloc_size = (size < 65536) ? 65536 : size;

  xrtDeviceHandle dev = get_xrt_device();
  if (!dev) return nullptr;

  /* Allocate buffer from XRT, memory group 0 (PSDDR / CMA) */
  xrtBufferHandle bo = xrtBOAlloc(dev, alloc_size, XRT_BO_FLAGS_CACHEABLE, 0);
  if (bo == nullptr) {
    /* Try without cacheable flag */
    bo = xrtBOAlloc(dev, alloc_size, XRT_BO_FLAGS_NONE, 0);
    if (bo == nullptr) {
      fprintf(stderr, "VTA: xrtBOAlloc failed for size %zu\n", alloc_size);
      return nullptr;
    }
  }

  /* Map to virtual address */
  void* buf = xrtBOMap(bo);
  if (buf == nullptr) {
    fprintf(stderr, "VTA: xrtBOMap failed\n");
    xrtBOFree(bo);
    return nullptr;
  }

  /* Get physical address */
  uint64_t phys = xrtBOAddress(bo);

  /* Zero-fill */
  memset(buf, 0, size);

  /* Sync to device after zeroing */
  xrtBOSync(bo, XCL_BO_SYNC_BO_TO_DEVICE, alloc_size, 0);

  /* Track allocation */
  {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    alloc_map[buf] = {bo, alloc_size, phys};
  }

  // fprintf(stderr, "VTA_ALLOC: size=%zu vaddr=%p phys=0x%08lx\n",
  //         alloc_size, buf, (unsigned long)phys);

  return buf;
}

void VTAMemFree(void* buf) {
  if (!buf) return;

  std::lock_guard<std::mutex> lock(alloc_mutex);
  auto it = alloc_map.find(buf);
  if (it == alloc_map.end()) {
    fprintf(stderr, "VTA: VTAMemFree on unknown buffer %p\n", buf);
    return;
  }

  xrtBOFree(it->second.bo_handle);
  alloc_map.erase(it);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  std::lock_guard<std::mutex> lock(alloc_mutex);
  auto it = alloc_map.find(buf);
  if (it != alloc_map.end()) {
    return (vta_phy_addr_t)it->second.phys_addr;
  }
  /* Try finding a containing allocation */
  it = find_containing_alloc(buf);
  if (it != alloc_map.end()) {
    uintptr_t offset = (uintptr_t)buf - (uintptr_t)it->first;
    return (vta_phy_addr_t)(it->second.phys_addr + offset);
  }
  fprintf(stderr, "VTA: VTAMemGetPhyAddr on unknown buffer %p\n", buf);
  return 0;
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // fprintf(stderr, "VTA_FLUSH: vaddr=%p phys=0x%08x size=%d\n", vir_addr, phy_addr, size);
  std::lock_guard<std::mutex> lock(alloc_mutex);
  auto it = find_containing_alloc(vir_addr);
  if (it != alloc_map.end()) {
    uintptr_t offset = (uintptr_t)vir_addr - (uintptr_t)it->first;
    xrtBOSync(it->second.bo_handle, XCL_BO_SYNC_BO_TO_DEVICE,
              (size_t)size, (size_t)offset);
  } else {
    fprintf(stderr, "VTA_FLUSH: WARNING no allocation found for %p\n", vir_addr);
  }
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // fprintf(stderr, "VTA_INVAL: vaddr=%p phys=0x%08x size=%d\n", vir_addr, phy_addr, size);
  std::lock_guard<std::mutex> lock(alloc_mutex);
  auto it = find_containing_alloc(vir_addr);
  if (it != alloc_map.end()) {
    uintptr_t offset = (uintptr_t)vir_addr - (uintptr_t)it->first;
    xrtBOSync(it->second.bo_handle, XCL_BO_SYNC_BO_FROM_DEVICE,
              (size_t)size, (size_t)offset);
  } else {
    fprintf(stderr, "VTA_INVAL: WARNING no allocation found for %p\n", vir_addr);
  }
}

/* ---- Register Access ---- */

void *VTAMapRegister(uint32_t addr) {
  uint32_t virt_base = addr & ~(getpagesize() - 1);
  uint32_t virt_offset = addr - virt_base;
  int mmap_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (mmap_fd < 0) {
    perror("VTA: open /dev/mem");
    return nullptr;
  }
  void* mapped = mmap(NULL,
              (VTA_IP_REG_MAP_RANGE + virt_offset),
              PROT_READ | PROT_WRITE,
              MAP_SHARED,
              mmap_fd,
              virt_base);
  close(mmap_fd);
  return mapped;
}

void VTAUnmapRegister(void *vta) {
  int status = munmap(vta, VTA_IP_REG_MAP_RANGE);
  assert(status == 0);
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

/* ---- Device Handle ---- */

class VTADevice {
 public:
  VTADevice() {
    vta_fetch_handle_ = VTAMapRegister(VTA_FETCH_ADDR);
    vta_load_handle_ = VTAMapRegister(VTA_LOAD_ADDR);
    vta_compute_handle_ = VTAMapRegister(VTA_COMPUTE_ADDR);
    vta_store_handle_ = VTAMapRegister(VTA_STORE_ADDR);
  }

  ~VTADevice() {
    VTAUnmapRegister(vta_fetch_handle_);
    VTAUnmapRegister(vta_load_handle_);
    VTAUnmapRegister(vta_compute_handle_);
    VTAUnmapRegister(vta_store_handle_);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    // fprintf(stderr, "VTA_RUN: insn_count=%u insn_phys=0x%08x\n", insn_count, insn_phy_addr);

    /*
     * Vivado HLS 2020.1 register layout — all pointers 32-bit.
     * Offsets match original VTA defines exactly.
     */

    // Fetch: insn_count @ 0x10, insns @ 0x18
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_COUNT_OFFSET, insn_count);
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_ADDR_OFFSET, insn_phy_addr);

    // Load: inputs @ 0x10, weights @ 0x18
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_INP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_WGT_ADDR_OFFSET, 0);

    // Compute: done_wr @ 0x10, uops @ 0x20, biases @ 0x28
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_WR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_UOP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_BIAS_ADDR_OFFSET, 0);

    // Store: outputs @ 0x10
    VTAWriteMappedReg(vta_store_handle_, VTA_STORE_OUT_ADDR_OFFSET, 0);

    // Clear any stale done_vld latch from a prior autorestart iteration —
    // reading the "valid" register on COR (clear-on-read) auto-clears it.
    // Without this, the post-start done_vld poll can see a latched value
    // from before the current run's FINISH instruction completes.
    (void)VTAReadMappedReg(vta_compute_handle_, 0x1c);

    // Start all modules with autorestart
    VTAWriteMappedReg(vta_fetch_handle_, 0x0, VTA_START);
    VTAWriteMappedReg(vta_load_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_store_handle_, 0x0, VTA_AUTORESTART);

    // Wait for fetch to finish dispatching all instructions to FIFOs.
    unsigned t;
    for (t = 0; t < wait_cycles; ++t) {
      uint32_t fetch_status = VTAReadMappedReg(vta_fetch_handle_, 0x0);
      if (fetch_status & 0x4) break;  // ap_idle
      std::this_thread::yield();
    }

    // Poll compute's done_vld (HLS 2020.1 function-return-value "valid" flag
    // at offset 0x1c). Asserted when compute has processed the FINISH insn
    // and latched the function-return-value; stays latched until the next
    // ack (DONE_WR write before the next Run). Load/store can't be polled
    // the same way — in AUTORESTART they block on empty FIFO reads and
    // ap_idle never asserts (confirmed session 6).
    static const uint32_t VTA_COMPUTE_DONE_VLD_OFFSET = 0x1c;
    unsigned t_done;
    bool done_seen = false;
    for (t_done = 0; t_done < wait_cycles; ++t_done) {
      uint32_t vld = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_VLD_OFFSET);
      if (vld & 0x1) { done_seen = true; break; }
      std::this_thread::yield();
    }

    if (done_seen) {
      // Compute has signaled done; small fixed drain covers the store DMA
      // pipeline tail (compute→store FIFO drain + AXI-MM write commit).
      usleep(50);
    } else {
      // Fallback: done_vld never latched (shouldn't happen for a well-formed
      // insn stream with a FINISH, but keep the old conservative drain as a
      // safety net). Scale with insn_count as before.
      unsigned drain_us = insn_count * 2;
      if (drain_us < 10) drain_us = 10;
      usleep(drain_us);
    }

    // fprintf(stderr, "VTA_DONE: fetch_iter=%u done_iter=%u done_seen=%d "
    //                 "fetch=0x%x load=0x%x compute=0x%x store=0x%x\n",
    //         t, t_done, done_seen ? 1 : 0,
    //         VTAReadMappedReg(vta_fetch_handle_, 0x0),
    //         VTAReadMappedReg(vta_load_handle_, 0x0),
    //         VTAReadMappedReg(vta_compute_handle_, 0x0),
    //         VTAReadMappedReg(vta_store_handle_, 0x0));
    (void)t_done;  /* only used in commented-out debug log above */

    // Stop autorestart on consumer modules
    VTAWriteMappedReg(vta_load_handle_, 0x0, 0x0);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, 0x0);
    VTAWriteMappedReg(vta_store_handle_, 0x0, 0x0);

    // Timeout if either fetch or compute poll ran out of budget.
    return (t < wait_cycles && done_seen) ? 0 : 1;
  }

 private:
  void* vta_fetch_handle_{nullptr};
  void* vta_load_handle_{nullptr};
  void* vta_compute_handle_{nullptr};
  void* vta_store_handle_{nullptr};
};

VTADeviceHandle VTADeviceAlloc() {
  return new VTADevice();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<VTADevice*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  /* Env-var-gated raw instruction-stream dump. Set VTA_DUMP_INSN_DIR=<dir>
   * (dir must exist) to capture every call's insn bytes as
   *   <dir>/call_<serial>.bin
   * with <dir>/index.txt logging "<serial>\t<insn_count>\t<insn_phy_addr>".
   * For byte-level diff between Python and C invocations of the same module.
   */
  const char *dump_dir = getenv("VTA_DUMP_INSN_DIR");
  if (dump_dir) {
    static std::atomic<uint32_t> call_serial{0};
    uint32_t serial = call_serial.fetch_add(1);

    /* Reverse lookup: find the BO containing insn_phy_addr. The existing
     * alloc_map is keyed by virtual address, so we scan. Each BO's
     * (phys_addr, size) gives its physical extent. */
    void *virt = nullptr;
    xrtBufferHandle bo_handle = nullptr;
    size_t offset_in_bo = 0;
    {
      std::lock_guard<std::mutex> lock(alloc_mutex);
      for (auto &kv : alloc_map) {
        uint64_t base = kv.second.phys_addr;
        uint64_t end  = base + kv.second.size;
        if (base <= (uint64_t)insn_phy_addr && (uint64_t)insn_phy_addr < end) {
          offset_in_bo = (uint64_t)insn_phy_addr - base;
          virt = (char*)kv.first + offset_in_bo;
          bo_handle = kv.second.bo_handle;
          break;
        }
      }
    }

    if (virt) {
      /* VTAGenericInsn is 16 bytes (128-bit VTA ISA insn). */
      size_t nbytes = (size_t)insn_count * 16;
      /* CommandQueue already flushed these bytes to the device via
       * AutoReadBarrier before calling us. The explicit BO_TO_DEVICE sync
       * here is defensive: ensures CPU-visible bytes match what the device
       * will read. */
      xrtBOSync(bo_handle, XCL_BO_SYNC_BO_TO_DEVICE, nbytes, offset_in_bo);

      char path[256];
      snprintf(path, sizeof(path), "%s/call_%04u.bin", dump_dir, serial);
      FILE *f = fopen(path, "wb");
      if (f) { fwrite(virt, 1, nbytes, f); fclose(f); }

      snprintf(path, sizeof(path), "%s/index.txt", dump_dir);
      FILE *idx = fopen(path, "a");
      if (idx) {
        fprintf(idx, "%u\t%u\t0x%08x\n",
                serial, insn_count, (uint32_t)insn_phy_addr);
        fclose(idx);
      }
    }
  }

  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}
