// dpu_c_runner.cpp — harness-symmetric DPU CNN/MLP runner (A1 / C1).
//
// Mirrors run_dpu_benchmark (benchmark.py) but runs the whole per-image loop in
// compiled code: one VART execute_async+wait per image, timed as a tight C loop
// (no per-image Python / pybind crossings). The input is the EXACT float32 NHWC
// tensor Python feeds VART, precomputed by board/dump_dpu_inputs.py and loaded
// from a .bin — NO quantization or normalization is done here (the DPU does the
// float->fix internally, same as the Python path).
//
// Build (host cross-compile against the card-pulled sysroot):
//   CXX=/tools/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
//   SYS=/home/samu/dev/CEN571-final/dpu_sysroot
//   $CXX -O2 -std=c++17 board/dpu_c_runner.cpp -o board/dpu_c_runner \
//       --sysroot=$SYS -I$SYS/usr/include -L$SYS/usr/lib \
//       -Wl,-rpath-link,$SYS/usr/lib -Wl,--allow-shlib-undefined \
//       -lvart-runner -lvart-util -lxir -lunilog -lglog -lprotobuf
//
// Emits a JSON close to run_dpu_benchmark's schema (toolchain "dpu_c",
// runs[].t_start/t_end epoch doubles) so merge_power.py can align FNB58 power.
// Cannot be run on the host (no DPU); on-board smoke test deferred.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>  // pulls in vart/tensor_buffer.hpp

// Declared in vart/runner_helper.hpp; forward-declared here to avoid that
// header's heavy transitive includes. Resolved at link time from libvart-*.
namespace vart {
std::unique_ptr<TensorBuffer> alloc_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor);
}

static double now_epoch() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

// ISO-8601 local timestamp for config.timestamp (schema parity with
// benchmark.py's datetime.now().isoformat()).
static std::string iso_now() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  return std::string(buf);
}

static std::string get_arg(int argc, char** argv, const std::string& key,
                           const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return def;
}

// Minimal "find the integer after a key" reader for the sidecar JSON.
static long json_int(const std::string& s, const std::string& key) {
  auto p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return -1;
  p = s.find_first_of("-0123456789", p + key.size() + 2);
  if (p == std::string::npos) return -1;
  return std::strtol(s.c_str() + p, nullptr, 10);
}

template <typename T>
static int argmax(const T* p, long n) {
  int best = 0;
  T bv = p[0];
  for (long i = 1; i < n; ++i)
    if (p[i] > bv) { bv = p[i]; best = (int)i; }
  return best;
}

int main(int argc, char** argv) {
  const std::string model   = get_arg(argc, argv, "--model");
  const std::string inputs  = get_arg(argc, argv, "--inputs");
  const std::string labels  = get_arg(argc, argv, "--labels");
  const std::string out     = get_arg(argc, argv, "--out", "dpu_c_result.json");
  const int runs            = std::stoi(get_arg(argc, argv, "--runs", "5"));
  const std::string shape   = get_arg(argc, argv, "--shape");     // "N,H,W,C"
  std::string sidecar       = get_arg(argc, argv, "--sidecar");
  if (model.empty() || inputs.empty() || labels.empty()) {
    std::cerr << "usage: " << argv[0]
              << " --model M.xmodel --inputs X.bin --labels L.bin "
                 "[--shape N,H,W,C | --sidecar X.json] [--runs R] [--out O.json]\n";
    return 1;
  }

  // dataset label from the input filename, for config.dataset parity with
  // benchmark.py (read by the analysis scripts; merge_power ignores it).
  const std::string dataset =
      inputs.find("mnist") != std::string::npos   ? "mnist"
      : inputs.find("cifar") != std::string::npos ? "cifar10"
      : "";

  // ---- shape: --shape, else sidecar (explicit or X.bin -> X.json) ----
  long N = -1, H = -1, W = -1, C = -1;
  if (!shape.empty()) {
    std::replace(const_cast<std::string&>(shape).begin(),
                 const_cast<std::string&>(shape).end(), ',', ' ');
    std::istringstream is(shape); is >> N >> H >> W >> C;
  } else {
    if (sidecar.empty()) {
      sidecar = inputs;
      auto dot = sidecar.rfind(".bin");
      if (dot != std::string::npos) sidecar = sidecar.substr(0, dot) + ".json";
    }
    std::ifstream js(sidecar);
    if (!js) { std::cerr << "ERROR: cannot read sidecar " << sidecar << "\n"; return 1; }
    std::stringstream ss; ss << js.rdbuf();
    const std::string j = ss.str();
    N = json_int(j, "count"); H = json_int(j, "H");
    W = json_int(j, "W");     C = json_int(j, "C");
  }
  if (N <= 0 || H <= 0 || W <= 0 || C <= 0) {
    std::cerr << "ERROR: bad shape (N=" << N << " H=" << H << " W=" << W
              << " C=" << C << ")\n"; return 1;
  }
  const long per_img = H * W * C;
  std::cout << "shape: " << N << "x" << H << "x" << W << "x" << C
            << " (per_image=" << per_img << " floats)\n";

  // ---- load precomputed float32 inputs + labels ----
  std::vector<float> in_data((size_t)N * per_img);
  {
    std::ifstream f(inputs, std::ios::binary);
    if (!f) { std::cerr << "ERROR: cannot open " << inputs << "\n"; return 1; }
    f.read(reinterpret_cast<char*>(in_data.data()),
           (std::streamsize)in_data.size() * sizeof(float));
    if (!f) { std::cerr << "ERROR: short read on " << inputs << "\n"; return 1; }
  }
  std::vector<uint8_t> lbl((size_t)N);
  {
    std::ifstream f(labels, std::ios::binary);
    if (!f) { std::cerr << "ERROR: cannot open " << labels << "\n"; return 1; }
    f.read(reinterpret_cast<char*>(lbl.data()), (std::streamsize)N);
  }

  // ---- graph -> DPU subgraph -> runner (mirrors benchmark.py:run_dpu_benchmark) ----
  auto graph = xir::Graph::deserialize(model);
  auto root = graph->get_root_subgraph();
  const xir::Subgraph* dpu_sg = nullptr;
  for (auto* sg : root->children_topological_sort()) {
    if (sg->has_attr("device") && sg->get_attr<std::string>("device") == "DPU") {
      dpu_sg = sg; break;
    }
  }
  if (!dpu_sg) { std::cerr << "ERROR: no DPU subgraph in " << model << "\n"; return 2; }
  std::cout << "DPU subgraph: " << dpu_sg->get_name() << "\n";

  auto runner = vart::Runner::create_runner(dpu_sg, "run");
  auto in_tensors  = runner->get_input_tensors();
  auto out_tensors = runner->get_output_tensors();
  const xir::Tensor* it = in_tensors[0];
  const xir::Tensor* ot = out_tensors[0];
  const long in_elems  = (long)it->get_element_num();
  const long out_elems = (long)ot->get_element_num();
  const size_t out_bytes = (size_t)ot->get_data_size();

  if (in_elems != per_img) {
    std::cerr << "ERROR: input tensor elems " << in_elems
              << " != per_image " << per_img << "\n"; return 2;
  }
  const bool out_is_float = (out_bytes == (size_t)out_elems * sizeof(float));
  std::cout << "input tensor: float32 x" << in_elems
            << " | output: " << out_elems
            << (out_is_float ? " (float32)" : " (int8)") << "\n";

  // Float passthrough: present VART a FLOAT input TensorBuffer (same name + shape
  // as the runner's int8 input tensor `it`). The DPU-DDR runner's
  // copy_data_for_input auto-converts float->int8 with the device tensor's
  // fix_point — the same path the Python vart binding uses (array_to_tensor_buffer).
  // No caller-side quant, no fix_point attr here. `fin` must outlive the runner.
  auto fin = xir::Tensor::create(it->get_name(), it->get_shape(),
                                 xir::DataType{xir::DataType::FLOAT, 32});
  auto in_tb  = vart::alloc_cpu_flat_tensor_buffer(fin.get());
  auto out_tb = vart::alloc_cpu_flat_tensor_buffer(ot);
  uint64_t in_addr, out_addr;
  size_t in_sz, out_sz;
  std::tie(in_addr, in_sz)   = in_tb->data(std::vector<int32_t>(it->get_shape().size(), 0));
  std::tie(out_addr, out_sz) = out_tb->data(std::vector<int32_t>(ot->get_shape().size(), 0));

  std::vector<vart::TensorBuffer*> ins{in_tb.get()}, outs{out_tb.get()};

  // One inference; returns argmax(pred). only_engine=true times nothing here —
  // callers wrap the execute+wait when they want the isolated engine latency.
  auto infer = [&](long idx) -> int {
    std::memcpy(reinterpret_cast<void*>(in_addr),
                &in_data[(size_t)idx * per_img], (size_t)per_img * sizeof(float));
    in_tb->sync_for_write(0, in_sz);
    auto v = runner->execute_async(ins, outs);
    runner->wait((int)v.first, -1);
    out_tb->sync_for_read(0, out_sz);
    if (out_is_float)
      return argmax(reinterpret_cast<float*>(out_addr), out_elems);
    return argmax(reinterpret_cast<int8_t*>(out_addr), out_elems);
  };

  // ---- idle window: 5 s, no inference, so the external FNB58 meter captures an
  // idle baseline. merge_power slices [idle.t_start, idle.t_end] for idle power
  // (hence dynamic power). Mirrors benchmark.py measure_idle — the piece the C
  // path was missing ("No idle timestamps in benchmark JSON").
  const double idle_t_start = now_epoch();
  std::this_thread::sleep_for(std::chrono::seconds(5));
  const double idle_t_end = now_epoch();

  // ---- warmup (10, discarded) ----
  for (long i = 0; i < std::min<long>(10, N); ++i) infer(i);

  // ---- measured runs ----
  std::ostringstream runs_json;
  std::vector<double> fps_list;
  double acc_last = 0.0;
  for (int r = 0; r < runs; ++r) {
    long correct = 0;
    const double t_start = now_epoch();
    for (long i = 0; i < N; ++i)
      if (infer(i) == (int)lbl[i]) ++correct;
    const double t_end = now_epoch();
    const double secs = t_end - t_start;
    const double fps = (double)N / secs;
    const double acc = 100.0 * (double)correct / (double)N;
    acc_last = acc; fps_list.push_back(fps);
    if (r) runs_json << ",\n";
    runs_json << "    {\"run\": " << (r + 1)
              << ", \"t_start\": " << std::fixed << t_start
              << ", \"t_end\": " << t_end
              << ", \"accuracy\": " << acc
              << ", \"time_s\": " << secs
              << ", \"throughput_fps\": " << fps
              << ", \"latency_ms\": " << (1000.0 / fps)
              << ", \"avg_power_w\": null"
              << ", \"energy_total_j\": null"
              << ", \"energy_per_image_mj\": null"
              << ", \"power_samples\": 0"
              << ", \"sysmon\": null}";
    std::cout << "run " << (r + 1) << ": " << acc << "% , " << (long)fps
              << " FPS\n";
  }

  // ---- C1 single-shot: K=200, engine-only AND all-in from the same calls ----
  // engine = execute_async+wait (the VART call; includes the in-hardware input
  //   quant, which is inseparable from it).
  // all-in = memcpy-in + execute + wait + argmax (input image -> prediction
  //   out), matching the Python DPU all-in boundary.
  const int K = 200;
  std::vector<double> ss_engine(K), ss_allin(K);
  for (int k = 0; k < K; ++k) {
    const long idx = k % N;
    const auto a0 = std::chrono::steady_clock::now();
    std::memcpy(reinterpret_cast<void*>(in_addr),
                &in_data[(size_t)idx * per_img], (size_t)per_img * sizeof(float));
    in_tb->sync_for_write(0, in_sz);
    const auto e0 = std::chrono::steady_clock::now();
    auto v = runner->execute_async(ins, outs);
    runner->wait((int)v.first, -1);
    const auto e1 = std::chrono::steady_clock::now();
    out_tb->sync_for_read(0, out_sz);
    if (out_is_float) (void)argmax(reinterpret_cast<float*>(out_addr), out_elems);
    else (void)argmax(reinterpret_cast<int8_t*>(out_addr), out_elems);
    const auto a1 = std::chrono::steady_clock::now();
    ss_engine[k] = std::chrono::duration<double, std::milli>(e1 - e0).count();
    ss_allin[k]  = std::chrono::duration<double, std::milli>(a1 - a0).count();
  }
  auto median_of = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double ss_engine_median = median_of(ss_engine);
  const double ss_allin_median  = median_of(ss_allin);

  // ---- aggregate ----
  double fps_mean = 0, lat_mean = 0;
  for (double f : fps_list) { fps_mean += f; lat_mean += 1000.0 / f; }
  fps_mean /= fps_list.size(); lat_mean /= fps_list.size();
  double fps_var = 0, lat_var = 0;
  for (double f : fps_list) {
    fps_var += (f - fps_mean) * (f - fps_mean);
    lat_var += (1000.0 / f - lat_mean) * (1000.0 / f - lat_mean);
  }
  fps_var = std::sqrt(fps_var / fps_list.size());
  lat_var = std::sqrt(lat_var / fps_list.size());

  // ---- emit JSON ----
  auto join = [](const std::vector<double>& v) {
    std::ostringstream s;
    for (size_t k = 0; k < v.size(); ++k) { if (k) s << ", "; s << v[k]; }
    return s.str();
  };
  std::ofstream o(out);
  o << std::fixed;
  o << "{\n  \"config\": {\n"
    << "    \"toolchain\": \"dpu_c\",\n"
    << "    \"model_path\": \"" << model << "\",\n"
    << "    \"dataset\": \"" << dataset << "\",\n"
    << "    \"batch_size\": 1,\n"
    << "    \"num_runs\": " << runs << ",\n"
    << "    \"num_images\": " << N << ",\n"
    << "    \"image_shape\": [" << H << ", " << W << ", " << C << "],\n"
    << "    \"dpu_input_shape\": [1, " << H << ", " << W << ", " << C << "],\n"
    << "    \"dpu_output_shape\": [1, " << out_elems << "],\n"
    << "    \"timestamp\": \"" << iso_now() << "\",\n"
    << "    \"board\": \"AUP-ZU3\",\n"
    << "    \"dpu\": \"DPUCZDX8G_ISA1_B512\",\n"
    << "    \"power_method\": \"none\",\n"
    << "    \"inputs\": \"" << inputs << "\",\n"
    << "    \"single_shot_latency_ms\": " << ss_allin_median << ",\n"
    << "    \"single_shot_latency_allin_ms\": " << ss_allin_median << ",\n"
    << "    \"single_shot_latency_allin_ms_values\": [" << join(ss_allin) << "],\n"
    << "    \"single_shot_latency_engine_ms\": " << ss_engine_median << ",\n"
    << "    \"single_shot_latency_engine_ms_values\": [" << join(ss_engine) << "]\n"
    << "  },\n"
    << "  \"idle\": {\n"
    << "    \"t_start\": " << idle_t_start << ",\n"
    << "    \"t_end\": " << idle_t_end << ",\n"
    << "    \"power\": {\"mean\": null, \"std\": null, \"n_samples\": 0},\n"
    << "    \"sysmon\": {\"temp_ps_c\": null, \"temp_pl_c\": null, "
       "\"vccint_v\": null, \"n_samples\": 0}\n"
    << "  },\n"
    << "  \"runs\": [\n" << runs_json.str() << "\n  ],\n"
    << "  \"summary\": {\n"
    << "    \"accuracy\": " << acc_last << ",\n"
    << "    \"throughput_fps_mean\": " << fps_mean << ",\n"
    << "    \"throughput_fps_std\": " << fps_var << ",\n"
    << "    \"latency_ms_mean\": " << lat_mean << ",\n"
    << "    \"latency_ms_std\": " << lat_var << ",\n"
    << "    \"idle_power_w\": null,\n"
    << "    \"idle_power_std\": null,\n"
    << "    \"idle_temp_pl_c\": null,\n"
    << "    \"avg_power_w_mean\": null,\n"
    << "    \"avg_power_w_std\": null,\n"
    << "    \"dynamic_power_w\": null,\n"
    << "    \"energy_per_image_mj_mean\": null,\n"
    << "    \"energy_per_image_mj_std\": null,\n"
    << "    \"sysmon\": null\n"
    << "  }\n}\n";
  o.close();
  std::cout << "wrote " << out << "  (single_shot_allin=" << ss_allin_median
            << " ms, engine=" << ss_engine_median
            << " ms, fps_mean=" << (long)fps_mean << ")\n";
  return 0;
}
