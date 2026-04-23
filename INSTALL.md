# Install

Prerequisites:

- Apple Silicon Mac (M1/M2/M3/M4), macOS 26+ (should work on 14+)
- Xcode Command Line Tools installed (`xcode-select --install`)
- Homebrew

## 1. Build tools

```bash
brew install python@3.12 ffmpeg cmake ninja ccache
```

Verify Python 3.12 is on PATH:

```bash
python3.12 --version    # -> 3.12.x
```

## 2. Clone this repo somewhere

```bash
cd ~/Documents/projects
git clone https://github.com/zbiles/applio-apple-silicon.git
```

## 3. Build patched PyTorch (~15 min first time)

```bash
cd ~/Documents/projects
git clone --depth 1 --branch v2.11.0 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive --depth 1
git apply ~/Documents/projects/applio-apple-silicon/pytorch-2.11.0.patch
~/Documents/projects/applio-apple-silicon/build-pytorch.sh
```

The wheel will be written to `/tmp/pytorch-wheels/torch-2.11.0-cp312-cp312-macosx_*_arm64.whl`.

Troubleshooting:
- `fatal error: 'string' file not found` during compile → your CLT install is missing `/Library/Developer/CommandLineTools/usr/include/c++/v1/`; reinstall CLT (`sudo rm -rf /Library/Developer/CommandLineTools && sudo xcode-select --install`) or the build script will fall back to the SDK copy.
- `Compatibility with CMake < 3.5 has been removed` → the build script sets `CMAKE_POLICY_VERSION_MINIMUM=3.5` to work around PyTorch's bundled protobuf. Re-run the script.
- `Could not compile universal protoc` → make sure `CMAKE_OSX_ARCHITECTURES` is **unset**. The build script handles this.

## 4. Install Applio

```bash
cd ~/Documents/projects
git clone https://github.com/IAHispano/Applio.git
cd Applio
chmod +x run-install.sh
./run-install.sh
```

This creates `.venv` with stock PyTorch 2.7.1. We're about to replace it.

## 5. Install the patched PyTorch into Applio's venv

```bash
cd ~/Documents/projects/Applio
.venv/bin/pip install --force-reinstall --no-deps \
    /tmp/pytorch-wheels/torch-2.11.0-cp312-cp312-macosx_*_arm64.whl
# torchaudio / torchvision must match the torch version
.venv/bin/pip install --upgrade torchaudio torchvision
```

## 6. Apply the Applio patch

```bash
cd ~/Documents/projects/Applio
git apply ~/Documents/projects/applio-apple-silicon/applio.patch
```

## 7. Train

```bash
cd ~/Documents/projects/Applio
./run-applio.sh
# Applio UI opens at http://127.0.0.1:6969
```

In the UI:

1. Use the Train tab's Preprocess → Extract Features → Start Training flow.
2. You should see:

   ```
   MPS workaround: m_source pinned to CPU and frozen.
   Starting training...
   0%|          | 0/179 [00:00<?, ?it/s]
   ```

3. After the first epoch the progress bar continues and TensorBoard logs show `loss/g/total` descending. Launch TensorBoard in a second terminal:

   ```bash
   ./run-tensorboard.sh
   # http://localhost:6870
   ```

## Verifying the patches are in effect

```bash
~/Documents/projects/Applio/.venv/bin/python -c "
import torch
assert torch.__version__ == '2.11.0'
assert torch.backends.mps.is_available()
x = torch.randn(4, 32000, device='mps')
s = torch.stft(x, n_fft=2048, hop_length=1000, win_length=2048,
               window=torch.hann_window(2048, device='mps'), return_complex=True)
print('OK: torch', torch.__version__, 'stft shape', s.shape)
"
```

Should print `OK: torch 2.11.0 stft shape torch.Size([4, 1025, 33])` with no warnings.
