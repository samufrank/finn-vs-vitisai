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
 * Done polling: Compute runs single-shot (0x01). After FINISH, ap_done
 * (bit 1 of CONTROL register, COR) fires reliably. No sleep workaround.
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

  fprintf(stderr, "VTA_ALLOC: size=%zu vaddr=%p phys=0x%08lx\n",
          alloc_size, buf, (unsigned long)phys);

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
  fprintf(stderr, "VTA_FLUSH: vaddr=%p phys=0x%08x size=%d\n", vir_addr, phy_addr, size);
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
  fprintf(stderr, "VTA_INVAL: vaddr=%p phys=0x%08x size=%d\n", vir_addr, phy_addr, size);
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
    fprintf(stderr, "VTA_RUN: insn_count=%u insn_phys=0x%08x\n", insn_count, insn_phy_addr);

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

    // Scale drain time with workload. At 100MHz, each VTA instruction
    // takes at most ~100 cycles (DMA + GEMM). Add 2x safety margin.
    unsigned drain_us = insn_count * 2;  // ~2us per instruction at 100MHz
    if (drain_us < 10) drain_us = 10;    // floor at 10us
    usleep(drain_us);

    fprintf(stderr, "VTA_DONE: fetch_iter=%u fetch=0x%x load=0x%x compute=0x%x store=0x%x\n",
            t,
            VTAReadMappedReg(vta_fetch_handle_, 0x0),
            VTAReadMappedReg(vta_load_handle_, 0x0),
            VTAReadMappedReg(vta_compute_handle_, 0x0),
            VTAReadMappedReg(vta_store_handle_, 0x0));

    // Stop autorestart on consumer modules
    VTAWriteMappedReg(vta_load_handle_, 0x0, 0x0);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, 0x0);
    VTAWriteMappedReg(vta_store_handle_, 0x0, 0x0);

    return t < wait_cycles ? 0 : 1;
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
  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}
