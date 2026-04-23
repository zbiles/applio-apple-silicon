# applio-apple-silicon

Patches that let you train [Applio](https://github.com/IAHispano/Applio) (RVC voice conversion) on Apple Silicon (MPS) without crashing.

Without these patches, the backward pass aborts the process with:

```
/AppleInternal/…/MPSCommandBufferImageCache.mm:1420: failed assertion
`Failed to allocate private MTLBuffer for size 0'
```

This repo contains:

- **`pytorch-2.11.0.patch`** — two small fixes to PyTorch's MPS backend.
- **`applio.patch`** — workarounds in Applio's code for additional MPS edge cases.
- **`build-pytorch.sh`** — a Mac-friendly PyTorch build recipe.
- **`docs/root-causes.md`** — what each patch does and why.
- **`docs/upstream-issues.md`** — tracking for upstream fixes; you can drop these patches once those land.

## Status

- macOS 26 (Tahoe) on Apple Silicon (M-series)
- PyTorch 2.11.0 (built from source with the patch applied)
- Applio commit `0ff669cf` (MIT License) — other recent commits should apply cleanly
- Tested: v2 HiFi-GAN NSF generator, 9-disc MPD, 40 kHz sample rate, batch size 4, CPU-based preprocessing + feature extraction, MPS training

Not tested: refinegan / MRF variants, multi-GPU, mixed-precision (fp16/bf16), Windows, Linux, CUDA (these paths are untouched — they should still work).

## What's compromised

The `SourceModuleHnNSF` module (a `Linear(1,1)` + `Tanh` inside the NSF generator) is moved to CPU and frozen during training. Its 2 scalar parameters don't update. This is a workaround for an MPS command-buffer scheduling assertion; removing it when upstream PyTorch fixes the underlying issue is a one-line revert in `train.py`.

## Install

See [`INSTALL.md`](INSTALL.md) for a step-by-step recipe.

Quick version:

```bash
# 1. Build patched PyTorch
git clone --depth 1 --branch v2.11.0 https://github.com/pytorch/pytorch.git
cd pytorch && git submodule update --init --recursive --depth 1
git apply /path/to/applio-apple-silicon/pytorch-2.11.0.patch
/path/to/applio-apple-silicon/build-pytorch.sh

# 2. Install wheel into Applio venv
/path/to/Applio/.venv/bin/pip install --force-reinstall --no-deps \
    /tmp/pytorch-wheels/torch-2.11.0-*.whl

# 3. Apply Applio patch
cd /path/to/Applio
git apply /path/to/applio-apple-silicon/applio.patch

# 4. Train normally via Applio's UI
./run-applio.sh
```

## Reproducers

Without the patches, each of these crashes. With the patches applied, all succeed.

### STFT on MPS (fixed by `pytorch-2.11.0.patch`)

```python
import torch
x = torch.randn(4, 32000, device='mps')
s = torch.stft(x, n_fft=2048, hop_length=1000, win_length=2048,
               window=torch.hann_window(2048, device='mps'),
               return_complex=True)
# Stock torch 2.11.0: UserWarning about resize from [] then success.
# Under RVC training (backward): `Failed to allocate private MTLBuffer for size 0'.
```

### Applio training forward → backward

The full crash manifests inside `loss_gen_all.backward()` during the first
training step. See `docs/root-causes.md` for the detailed trace.

## Why not upstream PRs?

The right long-term home for these fixes is PyTorch and Applio. I filed
(linked in `docs/upstream-issues.md`) the relevant issues/reproducers. Pushing
an agent-authored PR at busy maintainers adds review burden without earning
trust; a standalone repo lets them evaluate on their own schedule and cherry-pick
what they want.

## License

MIT. See [`LICENSE`](LICENSE).

## Related

- [PyTorch](https://github.com/pytorch/pytorch) — BSD-3-Clause
- [Applio](https://github.com/IAHispano/Applio) — MIT
- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — original upstream of the model architecture
