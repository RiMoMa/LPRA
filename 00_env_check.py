"""Record the hardware/software facts every later measurement depends on.

Flags come from /proc/cpuinfo, i.e. what CPUID reports to the kernel at boot on
THIS machine -- not inferred from the marketing model name.
"""
import json, os, platform, pathlib
import onnxruntime as ort, onnx, torch

def read(p):
    try: return pathlib.Path(p).read_text().strip()
    except OSError: return None

flags = set()
for line in read("/proc/cpuinfo").splitlines():
    if line.startswith("flags"):
        flags = set(line.split(":")[1].split()); break
model = next(l.split(":")[1].strip() for l in read("/proc/cpuinfo").splitlines() if l.startswith("model name"))

# Group logical CPUs by the L3 they share = one CCD each on Zen 4.
ccds = {}
for c in range(os.cpu_count()):
    base = f"/sys/devices/system/cpu/cpu{c}"
    ccds.setdefault(read(f"{base}/cache/index3/shared_cpu_list"), read(f"{base}/cache/index3/size"))

env = {
    "cpu": model,
    "avx2": "avx2" in flags,
    "avx512f": "avx512f" in flags,
    "avx512_vnni": "avx512_vnni" in flags,   # VPDPBUSD: u8*s8 -> int32 accumulate
    "avx_vnni": "avx_vnni" in flags,
    "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    "epp": read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
    "ccds_l3": ccds,
    "python": platform.python_version(), "torch": torch.__version__,
    "onnx": onnx.__version__, "onnxruntime": ort.__version__,
    "ort_providers": ort.get_available_providers(),
}
print(json.dumps(env, indent=2))
pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("results/env.json").write_text(json.dumps(env, indent=2))
