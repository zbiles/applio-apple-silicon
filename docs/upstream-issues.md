# Upstream issues

Tracking for the bugs this repo patches. Once upstream merges a fix, the corresponding patch here can be dropped.

## PyTorch

### STFT / FFT resize on MPS → zero-buffer assertion

- **Issue:** https://github.com/pytorch/pytorch/issues/174003
- **Scope:** Reported as a `UserWarning` only. In RVC training contexts the warning escalates to a hard Metal abort.
- **Status (2026-04):** Open, triaged, no PR yet.
- **Relevant file:** `aten/src/ATen/native/mps/operations/FastFourierTransform.mm`

### MPS allocator returns nullptr for 0-byte requests

- **Issue:** (file new one, or piggy-back on #174003 — they share the same root symptom)
- **Scope:** `DataPtr allocate(size_t nbytes)` returns nullptr if `nbytes == 0`. Downstream MPS code assumes valid `id<MTLBuffer>`.
- **Relevant file:** `aten/src/ATen/mps/MPSAllocator.mm`

### `MPSCommandBufferImageCache` zero-buffer assertion from closed Apple code

- **Issue:** This one fires inside `MPSCommandBufferImageCache.mm:1420`, Apple's closed-source MPS framework. Not patchable from PyTorch; the right layer is file a Radar with Apple.
- **Workaround in this repo:** `applio.patch` avoids the code paths that trip it (see `docs/root-causes.md` §6).

## Applio

### Attention pad specs include trailing `(0,0)` no-ops

- **Project:** https://github.com/IAHispano/Applio
- **Symptom on MPS:** pad specs cross the 6-element boundary, triggering the buggy fallback path.
- **Fix:** strip trailing `(0,0)` pairs in `convert_pad_shape`.
- **Relevant file:** `rvc/lib/algorithm/commons.py`

### `slice_segments` crashes when slice overruns wave

- **Symptom:** `RuntimeError: The expanded size of the tensor (12800) must match the existing size (0) at non-singleton dimension 1` on certain batches.
- **Scope:** pre-existing, unrelated to MPS. Appears to be a long-standing Applio / RVC data-padding edge case.
- **Fix:** clamp `idx_str`/`idx_end`, copy only the intersecting range, leave the tail as the zero fill.
- **Relevant file:** `rvc/lib/algorithm/commons.py`

### Unconditional `dist.init_process_group` on non-CUDA devices

- **Symptom:** `AttributeError: module 'torch.distributed' has no attribute 'init_process_group'` on PyTorch builds with `USE_DISTRIBUTED=0`.
- **Scope:** cosmetic for stock PyTorch (where `dist.init_process_group` exists). Actual runtime behavior with `world_size=1` is unnecessary.
- **Fix:** gate the call on `device.type == "cuda"`.
- **Relevant file:** `rvc/train/train.py`

## Maintenance strategy

If you're cherry-picking: the PyTorch patch is the most durable (targets narrow, well-defined functions). The Applio patch is the most likely to rot as Applio evolves, especially the `train.py` hunks that anchor off surrounding context. If you have trouble applying after a newer Applio commit, regenerate by porting the listed changes file-by-file — none are large.
