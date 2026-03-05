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
- Warp-level primitives (__shfl_sync, __ballot_sync)
- Occupancy tuning (block size, register usage)

**Priority 3 - Fine-tuning (<20% impact)**:
- Instruction-level parallelism
- Mixed precision (FP16/TF32)
- Prefetching and double buffering

## Success Criteria

- Correctness: All verification checks must pass (atol=1e-2, rtol=1e-2)
- Once verification passes, run profiling once for measurement, output a summary, and STOP.
- Profiling is informational — there is no mandatory performance target.
