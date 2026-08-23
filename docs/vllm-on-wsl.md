# Running vLLM under WSL2

vLLM has no native Windows build, so on Windows it runs inside WSL2 and
AutoDistiller benchmarks it over HTTP. WSL2 forwards `localhost`, so the
benchmark can run from Windows against a server in WSL.

Four things bite on a fresh WSL2 Ubuntu. Each is recorded here with its cause,
because none of the error messages point at the fix.

## 0. Prerequisite: a C/C++ toolchain

```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

vLLM JIT-compiles Triton kernels and, on some paths, invokes `nvcc`, which needs
a host C++ compiler and libstdc++ headers. A minimal Ubuntu image has neither.

Pip-installable substitutes (`ziglang` as `cc`) get Triton as far as linking but
fail at `nvcc`, which needs `<new>` and the rest of the C++ standard headers.
There is no userspace-only path; this step needs root.

## 1. Pinned memory is disabled by default on WSL

```
RuntimeError: UVA is not available
```

vLLM's `CudaPlatform.is_pin_memory_available()` returns `False` whenever it
detects WSL, as a conservative default. vLLM's newer GPU worker allocates UVA
buffers, UVA requires pinned memory, so startup fails outright.

On a modern WSL2 kernel pinned memory works fine. Confirm and opt in:

```bash
python -c "import torch; print(torch.zeros(8, pin_memory=True).is_pinned())"   # True
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

## 2. CUDA_HOME is not set

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

No separate CUDA toolkit install is needed: the torch wheel already vendors one.

```bash
export CUDA_HOME="$(python -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent / "cu13")')"
export PATH="$CUDA_HOME/bin:$PATH"
```

Use the directory matching your torch CUDA version (`torch.version.cuda`).

## 3. ninja is missing

```
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

```bash
uv pip install ninja
```

## Putting it together

```bash
uv venv --python 3.12 ~/vllm-env
uv pip install --python ~/vllm-env/bin/python vllm ninja
```

```bash
CU=~/vllm-env/lib/python3.12/site-packages/nvidia/cu13
VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME=$CU PATH=$CU/bin:$PATH \
  ~/vllm-env/bin/vllm serve Qwen/Qwen3-0.6B \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.80
```

Then, from Windows or WSL:

```bash
uv run autodistiller benchmark --endpoint http://localhost:8000 --backend vllm
```

## Sizing for a small card

On 8 GiB, vLLM reports what it actually used at startup:

```
Free memory on device (6.83/7.96 GiB) on startup. Desired GPU memory utilization
is (0.8, 6.37 GiB). Actual usage is 1.68 GiB for consumed memory (weights +
non-torch), 0.27 GiB for peak activation, and 0.32 GiB for CUDAGraph memory.
```

Qwen3-0.6B at `--gpu-memory-utilization 0.80` leaves ~4.4 GiB of KV cache, which
is 41k tokens, about 10 concurrent requests at a 4096-token context. Lower the
utilization if the desktop also needs the GPU; raise it on a headless machine.

`--enforce-eager` skips CUDA graph capture and starts faster, but it measures a
configuration nobody deploys. Leave it off when benchmarking.
