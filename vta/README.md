## VTA Deployment Scripts

VTA overlay deployment on AUP-ZU3 (ZU3EG).

### Contents

- `test_vta_mlp_full.py` — MLP MNIST inference (INT8, manual TE)
- `test_vta_mnist.py` — MNIST 10K accuracy test
- `test_vta_verify.py` — GEMM verification suite
- `test_vta_transformer_gemm.py` — transformer GEMM dimension verification
- `int4_option_a/` — INT4 bitstream build (buffer halving + AXI narrowing)
- `configs/` — canonical VTA config files for INT8 and INT4-o8, switching script
- `transformer/` — RadioML transformer deployment (see transformer/README.md)
