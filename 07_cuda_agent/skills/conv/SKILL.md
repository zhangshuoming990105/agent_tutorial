---
name: conv
description: Convolution kernel optimization skill for AMD GPUs. Covers Conv2d, ConvTranspose2d, and depthwise separable convolution with practical guidance on when handwritten kernels can and cannot beat vendor libraries.
tools:
  - run_shell
  - read_file
  - write_file
triggers:
  - conv
  - convolution
  - conv2d
  - conv_transpose
  - depthwise
  - transpose
  - spatial
always_on: false
---

# Convolution Kernel Optimization — AMD K100 (HIP)

## When a handwritten kernel CAN beat PyTorch

| Operation | Input scale | Why | Expected speedup |
|---|---|---|---|
| Depthwise conv | Any | Memory-bound, vendor libs under-optimised for groups=IC | 1.2–3× |
| Pointwise (1×1) conv | IC/OC ≤ 64 | Small GEMM, launch overhead dominates vendor path | 1.0–1.5× |
| Fused depthwise+pointwise | Any | Eliminates intermediate tensor write | 1.5–2× |
| Conv2d with small spatial | ≤ 32×32 | Vendor libs target large spatial; overhead dominates | 1.0–1.3× |

## When a handwritten kernel CANNOT beat PyTorch

| Operation | Input scale | Why | Expected speedup |
|---|---|---|---|
| **Standard Conv2d / ConvTranspose2d** | **Large spatial (≥128²), C ≥ 32** | **MIOpen uses matrix hardware (MFMA); scalar ALU is 10–15× slower** | **0.05–0.15×** |
| Large GEMM (matmul) | M,N,K ≥ 1024 | rocBLAS uses MFMA GEMM micro-kernels | 0.05–0.15× |

**Key principle**: on AMD MI-series / K100, the Matrix Fused Multiply-Add (MFMA)
unit delivers ~15× higher FLOPS than scalar FMA.  Any operation that PyTorch
routes through rocBLAS or MIOpen will use MFMA; a handwritten scalar kernel
_cannot_ compete unless it also uses MFMA intrinsics (currently not available
via standard HIP on Hygon K100).

## ConvTranspose2d: Algorithmic Analysis

For ConvTranspose2d with stride=1, padding=0:

```
output[n, oc, oh, ow] = Σ_{ic, kh, kw} input[n, ic, oh-kh, ow-kw] × weight[ic, oc, kh, kw]
```

Where out-of-bounds input reads are 0 (gather pattern).

**Compute**: 2 × N × OC × OH × OW × IC × KH × KW FLOPs.
Example: N=8, IC=OC=64, OH≈OW≈514, KH=3, KW=7 → 366 GFLOPs.

**Arithmetic intensity**: 366 GFLOP / (67 MB input + 0.3 MB weight + 67 MB output) ≈ 2.7 FLOP/byte.
This is compute-bound, NOT memory-bound. The bottleneck is FLOPS throughput.

## Three kernel families tested (auto-tuned)

### Family A: Input-patch LDS, Weight from L2

```
Grid:  (⌈OW/TOW⌉, N×OH, ⌈OC/TOC⌉)
Block: (TOW, TOC)   e.g. (16,16) = 256 threads
LDS:   TIC × KH × (TOW+KW−1) floats per IC tile
```

- Input patch loaded into LDS, reused by TOC output channels.
- Weight read from L2 per thread: unique (ic, oc, kh, kw).
- **Problem**: same input patch reloaded for each OC tile (OC/TOC blocks).
- **Best time**: 169 ms (0.12×).

### Family B: All-OC fused, Input-patch LDS

```
Grid:  (⌈OW/TOW⌉, N×OH)   — NO OC dimension
Block: (TOW, OC=64) = TOW×64 threads
LDS:   IC_ALL × KH × (TOW+KW−1) floats (full IC patch)
```

- Full input patch (all IC channels) loaded once into LDS.
- All 64 OC threads read from the same LDS data → no OC-dimension redundancy.
- **Best time**: 166 ms (0.12×). **Winner** for large-spatial tasks.

### Family C: Weight-LDS, Input from global (cuda-l1 reference approach)

```
Grid:  (⌈OW/BW⌉, ⌈OH/BH⌉, N×OC)
Block: (BW, BH) = 256 threads, one thread per output pixel
LDS:   IC × KH × KW floats = 5.25 KB for IC=64
```

- Weight in LDS, input directly from global/L2.
- 2D spatial tiling: better locality for input reads.
- **Works well for small inputs** (128×128 → L2 hit rate high).
- **Fails for large inputs** (512×512 → input exceeds L2, HBM bandwidth bottleneck).
- **Best time**: 292 ms (0.07×). Worst for our problem.

## Tile size auto-tuning pattern

When uncertain about optimal tile sizes, **instantiate multiple template
configurations and benchmark all of them at first invocation**:

```cpp
template<int TILE_W, int TILE_H, int TILE_IC>
__global__ void my_kernel(...);

static const Config CFGS[] = {
    {16, 16, 16, "16x16 TIC16"},
    { 8, 32, 16, " 8x32 TIC16"},
    {32, 16, 16, "32x16 TIC16"},
    // ...
};

static int g_best = -1;

// On first call: warm up + time all, cache best
// On subsequent calls: dispatch to cached best
extern "C" void launch_kernel(...) {
    if (g_best < 0) g_best = auto_tune(...);
    dispatch(g_best, ...);
}
```

Use `hipEventCreate/hipEventRecord/hipEventElapsedTime` for GPU-side timing.
Run 3 reps after 1 warmup. Total tuning overhead: ~0.5–2 s (one-time cost).

## Practical recommendations for the agent

1. **Check the operation first**: if it's a standard Conv2d/ConvTranspose2d with
   C ≥ 32 and spatial ≥ 128, expect speedup ≤ 0.15×. Write a correct kernel,
   profile once, and stop. Do not spend steps trying to optimise further.

2. **Depthwise conv is the sweet spot**: one channel per group means vendor libs
   have less to optimise. A simple thread-per-output-pixel kernel with shared
   memory for the input patch typically wins.

3. **For ConvTranspose2d stride=1**: use the gather formulation (output pixel
   reads from input). Prefer Family B (all-OC fused, input in LDS) for large
   spatial dimensions.

4. **Always define `#define WARP_SIZE 64`** and use `__shfl_down(val, offset)`
   (no mask) on Hygon K100 / AMD GPUs.

5. **When in doubt, auto-tune tile sizes** rather than guessing. Template
   instantiation + runtime benchmarking is cheap and eliminates guesswork.
