# DPUCZDX8G on AUP-ZU3: Build and Deployment Guide

Target: DPUCZDX8G B512 on AUP-ZU3 (ZU3EG, xczu3eg-sfvc784-1-e, 8 GB DDR4)
Flow: Vivado 2024.1 block design → PetaLinux 2024.1 (Docker) → `/dev/dpu` kernel driver
No XRT, no xclbin, no ZOCL.

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Vivado | 2024.1 | Native Ubuntu 24.04 (warns but works) |
| PetaLinux | 2024.1 | Docker Ubuntu 22.04 (does not build on 24.04) |
| Vitis AI | v4.0 branch | Recipes, compiler, Docker image |
| DPU IP | DPUCZDX8G v4.0.0 | From standalone reference design download |
| VART | 3.5 (Vivado variant) | From v4.0 branch `src/petalinux_recipes/` |
| Board BSP | Real Digital AUP-ZU3 | github.com/RealDigitalOrg/aup-zu3-bsp |

## 1. Tool Installation

### Vivado 2024.1

Install the AMD Unified Installer with Vivado selected (not Vitis). Under device support, Zynq UltraScale+ MPSoC is the only required family. Install to `/tools/Xilinx/Vivado/2024.1/`.

Install board files:
```bash
git clone https://github.com/RealDigitalOrg/aup-zu3-bsp.git
cp -r aup-zu3-bsp/board-files/* /tools/Xilinx/Vivado/2024.1/data/boards/board_files/
```

### DPU IP

The DPUCZDX8G is not in the Vivado IP catalog. Download separately:
```bash
wget "https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G.tar.gz" -O DPUCZDX8G.tar.gz
tar xzf DPUCZDX8G.tar.gz
```

The IP repo path is the directory containing `component.xml` — in this case `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/`.

### Vitis AI v4.0

```bash
git clone --branch v4.0 --single-branch https://github.com/Xilinx/Vitis-AI.git vitis-ai-v4.0
```

Provides PetaLinux recipes (`src/petalinux_recipes/`) and the model compiler Docker image.

### PetaLinux 2024.1 Docker

PetaLinux 2024.1 requires Ubuntu 22.04. Build a Docker container:

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Phoenix

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y \
    gawk wget git diffstat unzip texinfo gcc build-essential \
    chrpath socat cpio python3 python3-pip python3-pexpect \
    xz-utils debianutils iputils-ping python3-git python3-jinja2 \
    libegl1-mesa libsdl1.2-dev pylint xterm python3-subunit \
    mesa-common-dev zstd liblz4-tool file locales \
    libncurses5-dev libncursesw5-dev libssl-dev \
    bc lz4 net-tools rsync iproute2 sudo \
    libtinfo5 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    dpkg-reconfigure dash

RUN useradd -m -s /bin/bash plnx && echo "plnx ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER plnx
WORKDIR /home/plnx

COPY --chown=plnx:plnx petalinux-v2024.1-*-installer.run /tmp/petalinux-installer.run

RUN chmod +x /tmp/petalinux-installer.run && \
    mkdir -p /home/plnx/petalinux/2024.1 && \
    /tmp/petalinux-installer.run -d /home/plnx/petalinux/2024.1 -y && \
    rm /tmp/petalinux-installer.run

RUN echo "source /home/plnx/petalinux/2024.1/settings.sh" >> /home/plnx/.bashrc

CMD ["/bin/bash"]
```

Build and test:
```bash
docker build -t petalinux-2024.1 .
docker run --rm -it petalinux-2024.1 bash -c "which petalinux-build"
```

Note: the `COPY` line requires `--chown=plnx:plnx` — PetaLinux refuses to run as root and the installer must be owned by the build user.

## 2. Vivado Block Design

### Project setup

```bash
source /tools/Xilinx/Vivado/2024.1/settings64.sh
vivado &
```

Create a new project targeting the AUP-ZU3 board. If the board is not listed, select part `xczu3eg-sfvc784-1-e` manually. Add the DPU IP repo path under Settings → IP → Repository.

### Block design components

The design requires:

- Zynq UltraScale+ PS (from board preset — configures DDR4 8 GB, MIO, clocks)
- DPUCZDX8G (B512 configuration — see below)
- Clocking wizard (300 MHz DPU clock from PS pl_clk0)
- Three proc_sys_reset blocks (one per clock domain: PS clock, 300 MHz DPU clock, 600 MHz DPU 2x clock)
- AXI interconnects for DPU ↔ PS memory paths
- Interrupt concat (DPU interrupt → PS pl_ps_irq0)

Clock wizard reset: tie the `reset` input to an `xlconstant` set to 0. Do not expose it as an external pin — an undriven reset holds the wizard in reset permanently.

### DPU configuration

ZU3EG resource budget: 70,560 LUTs / 432 BRAM18 / 360 DSPs.

B512 configuration (verified to fit):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | B512 | 54.77% LUT, 33.33% BRAM, 37.22% DSP |
| Cores | 1 | |
| RAM Usage | Low | |
| Channel Augmentation | On | |
| DepthwiseConv | On | |
| AveragePool | On | |
| Softmax | Off | Hardware softmax has significant resource and CMA cost |
| DSP Usage | High | ZU3EG has DSPs to spare at B512 |

B1024 may fit but was not tested. B4096 does not fit on the ZU3EG.

Run synthesis after placing the DPU and check utilization before proceeding to implementation.

### AXI connectivity

DPU ports:
- M_AXI_HP0, M_AXI_HP1, M_AXI_HP2 (data + instruction) → PS S_AXI_HP0/HP1/HP2_FPD
- S_AXI_CONTROL → PS M_AXI_HPM0_FPD (register access)
- Interrupt → PS IRQ via xlconcat

All three M_AXI ports must be mapped to the same DDR base address in the address editor. Mismatched address maps cause silent DPU failures.

The ZCU102/ZCU104 reference Tcl scripts in `DPUCZDX8G/prj/Vivado/scripts/` show the expected wiring pattern. Adapt port names for the ZU3EG PS configuration.

### Generate bitstream and export

1. Validate block design
2. Create HDL wrapper
3. Synthesize — verify utilization
4. Implement — verify timing (target: WNS > 0)
5. Generate bitstream
6. File → Export → Export Hardware (include bitstream)

Output: `dpu_wrapper.xsa` — this is the input to PetaLinux.

Verified timing: WNS 0.295 ns, WHS 0.010 ns at 300/600 MHz.

## 3. PetaLinux Image

All PetaLinux commands run inside the Docker container:
```bash
docker run -it --rm -v ~/dev/CEN571-final:/home/plnx/project petalinux-2024.1
```

### Create project from BSP

```bash
cd /home/plnx/project
petalinux-create project -s aup-zu3-bsp/sw/petalinux/petalinux-8GB.bsp --name aup-zu3-dpu
cd aup-zu3-dpu
```

### Import hardware

```bash
petalinux-config --get-hw-description=/home/plnx/project/aup_zu3_dpu/dpu_wrapper.xsa --silentconfig
```

### Fix BSP device tree

The Real Digital BSP's `system-user.dtsi` references IPs (I2C, SPI, etc.) that are not present in the DPU XSA. These references cause build errors. Strip all IP references that are not part of the DPU block design from `project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi`, keeping only the base PS configuration and adding the CMA reservation:

```dts
/ {
    reserved-memory {
        #address-cells = <2>;
        #size-cells = <2>;
        ranges;

        linux,cma {
            compatible = "shared-dma-pool";
            reusable;
            size = <0x0 0x20000000>;  /* 512 MB */
            alignment = <0x0 0x2000>;
            linux,cma-default;
        };
    };
};
```

### Add Vitis AI recipes

```bash
cp -r /home/plnx/project/vitis-ai-v4.0/src/petalinux_recipes/recipes-vitis-ai \
    project-spec/meta-user/
cp -r /home/plnx/project/vitis-ai-v4.0/src/petalinux_recipes/recipes-vai-kernel \
    project-spec/meta-user/
```

### Select the Vivado VART recipe

The default `vart_3.5.bb` recipe pulls in XRT as a dependency, which is incompatible with the Vivado flow. Replace it with the Vivado-specific recipe:

```bash
cd project-spec/meta-user/recipes-vitis-ai/vart/
rm vart_3.5.bb
mv vart_3.5_vivado.bb vart_3.5.bb
```

Verify: `grep -i xrt vart_3.5.bb` should return nothing. If XRT appears as a dependency, the wrong recipe is active and VART will fail at runtime.

### Enable DPU kernel driver

```bash
petalinux-config -c kernel
```

Navigate to Device Drivers → Misc devices → Xilinx Deep Learning Processing Unit (DPU) Driver. Enable it, save, and exit.

### Enable Vitis AI libraries in rootfs

```bash
echo "CONFIG_vitis-ai-library" >> project-spec/meta-user/conf/user-rootfsconfig
echo "CONFIG_vitis-ai-library-dev" >> project-spec/meta-user/conf/user-rootfsconfig
petalinux-config -c rootfs
```

Select vitis-ai-library under user packages. Save and exit.

### Build

```bash
petalinux-build
petalinux-package --boot --u-boot --fpga --force
```

Build takes 30–90 minutes. Output in `images/linux/`: `BOOT.BIN`, `image.ub`, `boot.scr`, `rootfs.ext4`.

## 4. SD Card and Verification

### Write SD card

Partition the SD card: ~200 MB FAT32 (boot) + remainder ext4 (rootfs).

```bash
sudo parted /dev/sdX mklabel msdos
sudo parted /dev/sdX mkpart primary fat32 1MiB 201MiB
sudo parted /dev/sdX mkpart primary ext4 201MiB 100%
sudo mkfs.vfat -F 32 /dev/sdX1
sudo mkfs.ext4 /dev/sdX2

sudo mount /dev/sdX1 /mnt
sudo cp images/linux/BOOT.BIN /mnt/
sudo cp images/linux/boot.scr /mnt/
sudo cp images/linux/image.ub /mnt/
sudo umount /mnt

sudo dd if=images/linux/rootfs.ext4 of=/dev/sdX2 bs=4M status=progress
sudo resize2fs /dev/sdX2
```

### Boot and verify

Set the board boot switch to SD, insert the card, connect USB-UART, and power on. Default login: `petalinux` (prompts for a new password on first boot — set to `zu3` for consistency with the project).

```bash
ls /dev/dpu                          # should exist
cat /proc/interrupts | grep dpu      # should show DPU interrupt
xdputil query                        # shows DPU arch, fingerprint, frequency
```

Record the fingerprint from `xdputil query` — it is required to compile models. For this build: `0x101000016010400`.

### DPU optimization

```bash
cd ~/dpu_sw_optimize/zynqmp/
./zynqmp_dpu_optimize.sh
```

Sets AXI QoS and outstanding command limits for DPU traffic.

## References

- AMD PG338 — DPUCZDX8G Product Guide (DPU configuration parameters, resource usage)
- [Vitis AI 3.0 system integration (Vivado flow)](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html)
- [DPU TRD for 2024.x (LogicTronix)](https://www.hackster.io/LogicTronix/vitis-ai-dpu-trd-for-2024-x-with-vivado-and-kria-mpsoc-703403)
- [Vitis AI 3.5 with PetaLinux 2024.2 (LogicTronix)](https://www.hackster.io/LogicTronix/vitis-ai-3-5-with-petalinux-2024-2-36c9f7)
- [Real Digital AUP-ZU3 BSP](https://github.com/RealDigitalOrg/aup-zu3-bsp)
