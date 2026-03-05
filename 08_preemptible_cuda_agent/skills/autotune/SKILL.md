---
name: autotune
description: Template-based kernel auto-tuning strategy. Instantiate multiple tile-size configurations at compile time, benchmark all at first invocation, cache and reuse the fastest. Use when optimal tile sizes are uncertain.
tools:
  - run_shell
  - read_file
  - write_file
triggers:
  - autotune
  - auto-tune
  - tile size
  - tiling
  - template
  - benchmark
  - config
always_on: false
---

# Auto-Tune Strategy: Template-Based Kernel Tuning

When you are uncertain about the best tile sizes, block dimensions, or other
kernel tuning parameters, **do not guess — instantiate several configurations
as C++ templates and let the GPU pick the fastest one at runtime**.

## When to Use

- Multiple plausible tile/block sizes and no clear winner from analysis
- Kernel performance is sensitive to occupancy, LDS usage, or register pressure
- The operation will be called many times (tuning overhead amortised)

## Implementation Pattern

### 1. Templatise the kernel

```cpp
template<int TILE_W, int TILE_H, int TILE_IC, bool USE_LDS_WEIGHT>
__global__ void my_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    /* ... */)
{
    constexpr int PATCH_W = TILE_W + KW - 1;
    __shared__ float smem[/* computed from template params */];
    // ... kernel body using TILE_W, TILE_H, etc. as compile-time constants
}
```

**Rules:**
- Template parameters must be compile-time constants → enables full unrolling
- Use `constexpr` for derived sizes (shared memory, loop bounds)
- Use `#pragma unroll` on inner loops over kernel spatial dims
- Keep template params to 2–4 dimensions to limit instantiation count

### 2. Define a config table

```cpp
struct Config {
    int tile_w, tile_h, tile_ic;
    bool use_lds_weight;
    const char* tag;   // human-readable name for logging
};

static const Config CONFIGS[] = {
    {16, 16, 16, false, "16x16 TIC16 L2W"},
    { 8, 32, 16, false, " 8x32 TIC16 L2W"},
    {32, 16,  8, true,  "32x16 TIC8  LDS"},
    // ... 8–15 configs is a good range
};
static const int N_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);
static int g_best_cfg = -1;   // cached after first auto-tune
```

**Guidelines for config selection:**
- Vary block size: 128, 256, 512, 1024 threads (must be multiple of 64 for AMD)
- Vary tile width: powers of 2 from 8 to 64
- Vary IC tile depth: 8, 16, 32
- Include at least one "boolean strategy" variant (e.g. weight in LDS vs L2)
- Stay within hardware limits: ≤ 1024 threads/block, ≤ 64 KB LDS/block

### 3. Write a dispatch function

```cpp
static void dispatch(int cfg,
    const float* inp, const float* wgt, float* out,
    int N, int H, int W, /* ... */
    hipStream_t stream)
{
    const auto& c = CONFIGS[cfg];
    // Switch on config index — each case launches the matching template
    switch (cfg) {
        case 0: { dim3 g(...); dim3 b(c.tile_w, c.tile_h);
                  my_kernel<16,16,16,false><<<g,b,0,stream>>>(inp,wgt,out,...); } break;
        case 1: { /* ... */ } break;
        // ...
    }
}
```

### 4. Auto-tune on first call

```cpp
static int auto_tune(const float* inp, const float* wgt, float* out,
                     int N, int H, int W, /* ... */)
{
    hipStream_t s; hipStreamCreate(&s);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    float best_ms = FLT_MAX;
    int best_i = 0;
    constexpr int WARMUP = 1, REPS = 3;

    for (int i = 0; i < N_CONFIGS; i++) {
        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            dispatch(i, inp, wgt, out, N, H, W, ..., s);
        }
        hipStreamSynchronize(s);

        // Timed runs
        hipEventRecord(t0, s);
        for (int r = 0; r < REPS; r++)
            dispatch(i, inp, wgt, out, N, H, W, ..., s);
        hipEventRecord(t1, s);
        hipStreamSynchronize(s);

        float ms = 0;
        hipEventElapsedTime(&ms, t0, t1);
        ms /= REPS;
        if (ms < best_ms) { best_ms = ms; best_i = i; }
    }

    hipEventDestroy(t0); hipEventDestroy(t1);
    hipStreamDestroy(s);
    return best_i;
}
```

### 5. Public entry point with caching

```cpp
extern "C" void launch_my_kernel(/* ... */, cudaStream_t stream)
{
    if (g_best_cfg < 0)
        g_best_cfg = auto_tune(inp, wgt, out, N, H, W, ...);
    dispatch(g_best_cfg, inp, wgt, out, N, H, W, ..., (hipStream_t)stream);
}
```

## Practical Notes

- **Tuning cost**: ~0.5–2 s for 10–15 configs with 3 reps each. One-time only.
- **Correctness**: all configs share the same algorithm, just different tile sizes.
  Verify with ANY config; the auto-tune only selects the fastest.
- **Compile time**: each template instantiation adds ~1–2 s to hipcc. Keep to
  ≤ 15 configs to stay under 30 s total compile.
- **AMD specifics**: block size must be a multiple of 64 (wavefront size).
  Configs with thread count > 1024 will silently fail — exclude them.
- **Printf in auto-tune**: print timing per config so the profiling output
  shows which config was selected. This helps diagnose performance issues.

## When NOT to use auto-tuning

- Simple elementwise / pointwise kernels: tile size barely matters
- Single clear optimal config from analysis (e.g. depthwise conv → 1 thread/pixel)
- The kernel runs only once (tuning overhead not amortised)
