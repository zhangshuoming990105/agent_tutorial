---
name: cuda
description: CUDA kernel development and optimization skill. Guides the agent through implementing, compiling, verifying, and profiling custom CUDA extensions for PyTorch models.
tools:
  - run_shell
  - read_file
  - write_file
  - list_directory
  - search_files
  - grep_text
triggers:
  - cuda
  - kernel
  - optimize
  - gpu
  - compile
  - profile
  - verify
  - model
  - performance
  - speedup
always_on: true
---

# CUDA Kernel Development Skill

You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension.

## ⚠️ Hardware: AMD GPU (ROCm / HIP backend)

This system uses an **AMD GPU** compiled via HIP. Critical differences from NVIDIA CUDA:

| Topic | NVIDIA CUDA | AMD HIP (this system) |
|---|---|---|
| Wavefront / warp size | 32 threads | **64 threads** |
| Warp shuffle | `__shfl_down_sync(0xFFFFFFFF, v, d)` | `__shfl_down(v, d)` (no mask) |
| Ballot | `__ballot_sync(0xFFFFFFFF, pred)` | `__ballot(pred)` (no mask) |
| Sync | `__syncthreads()` | same |

**Always define and use:**
```cpp
#define WARP_SIZE 64   // AMD wavefront = 64 threads, NOT 32
```

All warp-level reductions must loop or tile for 64 threads:
```cpp
// Correct warp reduction for AMD (64-lane wavefront)
for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
```

Block-level reduction with shared memory (accounting for 64-lane warp):
```cpp
__shared__ float smem[WARP_SIZE];   // one slot per warp
int lane = threadIdx.x % WARP_SIZE;
int wid  = threadIdx.x / WARP_SIZE;
// 1. warp-level reduce
for (int d = WARP_SIZE/2; d > 0; d >>= 1) val += __shfl_down(val, d);
// 2. first lane of each warp writes to smem
if (lane == 0) smem[wid] = val;
__syncthreads();
// 3. first warp reduces smem
val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem[lane] : 0.0f;
if (wid == 0) { for (int d = WARP_SIZE/2; d > 0; d >>= 1) val += __shfl_down(val, d); }
```

## Critical Restrictions

### Strictly Forbidden
- **NO torch operators in C++**: Never use `torch::*` or `torch::nn::functional::*` in binding.cpp or .cu files
- **NO torch operations in model_new.py**: Only tensor creation and your custom ops allowed
- **NO third-party libraries**: No cuBLAS, cuDNN, MIOpen, or any external library calls. Write raw CUDA kernels only.
- **NO modifications to utils/ directory**
- **NO modifications to binding.cpp or binding_registry.h**: These are fixed infrastructure

### Allowed Only
- **C++**: Raw CUDA kernels with `__global__` functions only. All math must be hand-written.
- **Python**: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
- **Memory**: torch::empty_like for allocation only
- **Focus**: Implement kernels in `kernels/` directory only

## Workspace Structure

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── utils/                # DO NOT modify - Compilation, verification and profiling tools
├── model.py              # DO NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops
```

### File Types
- **`.cu` files**: CUDA kernels with `__global__` functions (custom implementations)
- **`_binding.cpp` files**: PyTorch tensor handling and Python bindings

## Workflow

### Step 1: Read and Analyze model.py
Understand the PyTorch model's forward pass and plan which operations to implement as CUDA kernels.

### Step 2: Implement Kernels
Create paired files in `kernels/`:

**kernels/my_kernel.cu** - Pure CUDA kernel with C-interface launcher:
- Use `extern "C"` launcher function
- Accept `cudaStream_t` parameter
- Use `config` parameter for runtime tuning

**kernels/my_kernel_binding.cpp** - PyTorch binding:
- Use `#include <torch/types.h>` and `#include <torch/csrc/utils/pybind.h>` (NOT torch/extension.h)
- Validate inputs: is_cuda, is_contiguous, dtype, shape
- Get stream via `c10::cuda::getCurrentCUDAStream().stream()`
- Register with `REGISTER_BINDING(name, register_func)`

### Step 3: Write model_new.py
- `import cuda_extension`
- `ModelNew` must match `Model` constructor signature exactly
- Forward uses `cuda_extension.<op>()` calls only

### Step 4: Compile and Test
```bash
bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```

### Step 5: Iterate on Errors

**Compilation errors**: Check extern "C" declarations, include paths, syntax.
**Correctness failures**: Check boundary conditions (tid < size), synchronization, data types, memory alignment.
**Performance**: Apply optimizations in priority order below.

## Optimization Priorities

**Priority 1 - Algorithmic (>50% impact)**:
- Kernel fusion to reduce memory traffic
- Shared memory tiling for data reuse
- Memory coalescing for consecutive access

**Priority 2 - Hardware Utilization (20-50% impact)**:
- Vectorized loads (float2/float4)
- Warp-level primitives (use AMD HIP syntax — see hardware section above)
- Occupancy tuning (block size, register usage)

**Priority 3 - Fine-tuning (<20% impact)**:
- Instruction-level parallelism
- Mixed precision (FP16/TF32)
- Prefetching and double buffering

## Common Kernel Patterns

### Elementwise / pointwise
```cpp
// grid-stride loop with float4 vectorization
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
float4* in4  = reinterpret_cast<float4*>(input);
float4* out4 = reinterpret_cast<float4*>(output);
for (int i = idx; i < n/4; i += stride) {
    float4 v = in4[i];
    v.x = op(v.x); v.y = op(v.y); v.z = op(v.z); v.w = op(v.w);
    out4[i] = v;
}
// handle tail
```

### Reduction (sum, max, …)
Use the two-level warp→block pattern shown in the hardware section above.
Launch with `blockDim.x = 256` (4 warps of 64).

### GEMM (matrix multiply)
Use 2D shared-memory tiling. Recommended tile sizes for AMD:
```cpp
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
// Each thread computes a 4×4 output sub-tile (register blocking).
// Load TILE_M × TILE_K of A and TILE_K × TILE_N of B into shared memory.
// Inner loop over K in steps of TILE_K.
```
Aim for `blockDim = (16, 16)` → 256 threads = 4 wavefronts.

### 2D Convolution (depthwise)
```cpp
// One thread per output element, shared memory tile for the input patch.
// blockDim = (16, 16), each block covers a 16×16 spatial tile.
// Load input tile + halo into shared memory, then iterate over kernel.
```

### Prefix scan / cumsum (small N ≤ 1024)
Use a single block with work-efficient Blelloch scan in shared memory.
For large N, use a two-pass approach: per-block scan + inter-block prefix.

## Success Criteria

- Correctness: All verification checks must pass (atol=1e-2, rtol=1e-2).
- **Performance stopping rule**:
  - After the first profiling run: if `cuda_time ≤ torch_compile_time × 1.1` → output summary and **STOP**.
  - If `cuda_time > torch_compile_time × 1.1` (kernel is slower): make **exactly one more** rewrite using a fundamentally different strategy (e.g. switch from naive to tiled, or reduce register pressure), re-compile, re-verify, re-profile, then **STOP regardless** of the new result.
  - Never attempt more than two full compile→verify→profile cycles per run.
