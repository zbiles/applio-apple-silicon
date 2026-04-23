# Root causes

What each patch in this repo actually fixes, with references to the source lines.

## 1. `torch.stft` on MPS allocates an empty output tensor

**File:** `aten/src/ATen/native/mps/operations/FastFourierTransform.mm`

`_fft_r2c_mps` and `_fft_c2r_mps` allocate their output with `at::empty({}, ...)` (shape `[]`, zero elements) and then call the `_out` variant which calls `resize_output_symint(out, out_sizes)` to give it the correct shape. PyTorch emits this warning every time:

```
UserWarning: An output with one or more elements was resized since it had shape []
which does not match the required output shape [B, F, N].
```

On MPS, this zero-shape placeholder eventually reaches Apple's internal `MPSCommandBufferImageCache`, which asserts when asked to allocate a 0-byte scratch buffer. Under inference it's a warning; under training's backward pass it's fatal.

**Fix:** pre-compute the correct shape and allocate directly (`at::empty_symint(out_sizes, ...)`). Same math that `_out` was doing, just lifted up so the tensor is never zero-sized.

Upstream: see `docs/upstream-issues.md`.

## 2. `MPSAllocator` returns `nullptr` for zero-byte requests

**File:** `aten/src/ATen/mps/MPSAllocator.mm`

```cpp
DataPtr allocate(const size_t nbytes) override {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().malloc(nbytes, m_usage) : nullptr;
    ...
}
```

When `nbytes == 0`, this returns a tensor backed by `nullptr`. Later ops bind this to Metal command encoders, which can trip scheduling assertions.

**Fix:** always allocate at least 1 byte. The extra byte is wasted but guarantees a valid `id<MTLBuffer>` and makes downstream MPS code robust against zero-shape tensors.

## 3. Applio `convert_pad_shape` keeps trailing `(0,0)` pairs

**File:** `rvc/lib/algorithm/commons.py`

PyTorch's `F.pad(input, pad)` takes a tuple where pairs are applied from the last dim inward. Applio's attention code writes things like:

```python
F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
```

which flattens to `(0, 1, 0, 0, 0, 0, 0, 0)`. The trailing `(0, 0)` pairs are no-ops (pad outer dims by zero), but they push the pad spec beyond 6 elements, which triggers MPS's `constant_pad_nd` fallback path:

```
MPS: The constant padding of more than 3 dimensions is not currently supported natively.
It uses View Ops default implementation to run.
```

That fallback's backward is where the zero-buffer crash originates during training.

**Fix:** in `convert_pad_shape`, strip trailing `(0, 0)` pairs. `F.pad(x, (0, 1))` is mathematically identical to `F.pad(x, (0, 1, 0, 0, 0, 0, 0, 0))` but stays on the native MPS path.

## 4. Applio `slice_segments` crashes on edge cases

**File:** `rvc/lib/algorithm/commons.py`

When `ids_slice * hop_length + segment_size` overshoots `wave.size(-1)`, the slice produces a tensor with shape `[0]` that can't be assigned into the prebuilt `[C, segment_size]` zeros buffer. Manifests as:

```
RuntimeError: The expanded size of the tensor (12800) must match the existing size (0)
at non-singleton dimension 1.
```

This is a pre-existing Applio bug (not MPS-specific) that only shows up on some batches, depending on which samples get shuffled in.

**Fix:** clamp `idx_end` to `wave.size(-1)` and only copy the intersecting region; the rest stays as the zero fill. Graceful no-op when a slice lands entirely past the end.

## 5. `torch.distributed` required even for single-process MPS

**File:** `rvc/train/train.py`

Applio unconditionally calls `dist.init_process_group(...)` with `world_size=1` when not on CUDA. Our custom PyTorch build has `USE_DISTRIBUTED=0` (smaller wheel, no need for it on MPS), so this call raises `AttributeError`.

**Fix:** only call `init_process_group` when `device.type == "cuda"`. Applio's `DistributedBucketSampler` doesn't actually use any distributed APIs — just takes `num_replicas`/`rank` as integer args — so no other changes are needed.

## 6. `SourceModuleHnNSF` backward trips an MPS command-buffer assertion

**Files:** `rvc/lib/algorithm/generators/hifigan_nsf.py`, `rvc/train/train.py`

Even with the STFT/pad fixes, backward through the NSF module's `Linear(1,1)` + `Tanh` (fed from a `torch.no_grad()`-wrapped sine generator) triggers:

```
-[_MTLCommandBuffer addScheduledHandler:]:807: failed assertion
`Scheduled handler provided after commit call'
```

**Fix:** move `SourceModuleHnNSF` to CPU and freeze its 2 scalar parameters. `SourceModuleHnNSF.forward` is overridden so that when `_force_cpu=True`, inputs are transferred to CPU, the module runs, and outputs are transferred back. An `_apply` override prevents `model.to('mps')` from undoing the CPU pin. `train.py` sets the flag and freezes the params right after `net_g.to(device)`.

Trade-off: the 2 scalar params don't train. In practice the pretrained values are fine; the quality of learned voice comes from the ~tens-of-millions of other parameters in the generator.

## Sequence of failure modes (without patches)

Debugging the full stack showed each fix was necessary:

1. Training starts. First `F.pad >3 dims` warning fires from attention.
2. Forward pass completes. First net_d completes.
3. Second net_d completes. Loss computation (including mel via STFT) completes.
4. `loss_gen_all.backward()` runs.
5. Backward flows from `net_g.dec.conv_post` down through the generator.
6. Eventually reaches `m_source.l_linear` backward.
7. Crash: `Failed to allocate private MTLBuffer for size 0'`.

Each patch clears one barrier. Remove any single patch and the crash returns at a related point.
