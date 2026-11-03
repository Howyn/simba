# Negative-eigenvalue Mamba experiment on NTU60 X-Sub joint

This experiment keeps the original Simba baseline and changes only the real Mamba scan transition from `exp(delta*A)` to `2*exp(delta*A)-1`. Both configurations use the existing data, seed, augmentation and learning rate schedule. Training stops at 10 epochs, optionally continues to 20. Neither setting proves a 50-epoch result. The CUDA implementation needs verification on your AutoDL GPU before use; do not treat successful compilation as proof of correctness.

## Setup

Use the existing Linux AutoDL environment where the baseline Mamba extension and `causal-conv1d` are already installed. The CUDA toolkit (`nvcc`), PyTorch and the GPU driver must be compatible. The default NPZ is `/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz`; keep exactly the same file as the previous baseline. Edit both experimental YAMLs if the path differs. On the same GPU and environment for both groups:

```bash
git pull origin main
nvidia-smi
nvcc --version
python -c 'import torch, selective_scan_cuda; print(torch.__version__, torch.cuda.is_available(), selective_scan_cuda.__file__)'
python setup_negative_scan.py build_ext --inplace
python -c 'import negative_selective_scan_cuda as m; print(m.__file__)'
python -m unittest discover -s tests -p 'test_negative_scan.py' -v
```

**Stop if a build or any test fails.** The test compares forward and gradients against the reference recurrence, including a sequence crossing a scan chunk, then compares the fused Mamba path with the unfused negative CUDA path. The latter does not independently validate Mamba against a full Python reference. Neither test is a classification result. Only real A is supported; cached/step inference is intentionally unsupported.

Then inspect one real batch and all 10 Mamba layers:

```bash
python tools/inspect_negative_transitions.py
```

Check finite loss and gradients and report transition statistics. This diagnostic includes every resampled position because the model does not expose a padding mask. It disables random rotation and is a diagnostic, not a training accuracy measurement. Optional `--weights work_dir/.../runs-latest.pt` inspects a trained checkpoint. If VRAM is insufficient, pass `--batch-size 1`.

## First 10 epochs

```bash
bash run_negative_experiment.sh baseline 10
bash run_negative_experiment.sh negative 10
```

Each group starts from scratch in a separate `work_dir/ntu60/xsub/negative_eigen_*` directory; the launcher refuses to overwrite an existing directory. The negative launcher runs numerical checks before starting. Each evaluation writes accuracy, per-class outputs and TensorBoard logs. `runs-latest.pt` contains recovery state. `runs-best.pt` is selected by the existing evaluation loop. Compare epoch 10, entire curves, loss, stability, time and memory. Existing code evaluates the official X-Sub test split every epoch; repeatedly selecting experiments on it leaks test information. Use a fixed held-out training-subject validation split for selection if available; otherwise label results exploratory.

## Optional continuation to 20 epochs

Only continue if both runs finished at epoch 10, their latest checkpoints are intact and the comparison merits it:

```bash
bash run_negative_experiment.sh baseline 20
bash run_negative_experiment.sh negative 20
```

The launcher verifies the saved epoch and restores model, optimizer and RNG via `--strict-rng-resume True`, with persistent workers disabled. It does not promise bitwise equivalence of GPU execution. Do not change configs, data or environment between stages. The original milestone 40 remains unchanged; there is no LR decay during these 20 epochs. Repeating the study with another seed needs new, matching work directories and configurations. The scan adds no learnable weights, but runtime and memory overhead must be measured on the target GPU.

Reference: Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/file/5a0ce3abb720b740419e193c87afd080-Paper-Conference.pdf
