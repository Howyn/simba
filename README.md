# Simba: NTU RGB+D 60 X-Sub Joint baseline

This repository contains the NTU60 cross-subject joint-stream baseline for [Simba](https://arxiv.org/abs/2404.07645). The optional experimental architectures and other dataset configurations have been removed. The baseline retains the NTU joint-dimension fix and the existing AutoDL training settings.

## Environment

Use Linux with an NVIDIA GPU, a compatible CUDA toolkit and PyTorch installation. On an AutoDL instance with PyTorch already installed:

```bash
python -m pip install -r requirements-autodl.txt
python -m pip install -e . --no-build-isolation
```

The editable install builds or obtains the Mamba selective-scan CUDA extension. The PyTorch, CUDA, Triton and causal-conv1d versions must be compatible. The dataset and checkpoints are not included in Git.

## Data

Obtain the NTU RGB+D 60 skeleton data from the [dataset provider](https://rose1.ntu.edu.sg/dataset/actionRecognition/). Extract `nturgbd_skeletons_s001_to_s017.zip` under `/root/autodl-tmp/ntu60/nturgbd_raw` and prepare the X-Sub split:

```bash
python tools/prepare_ntu60_xsub.py
```

The default output is `/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz`. See `python tools/prepare_ntu60_xsub.py --help` for alternative paths, and update both `data_path` entries in `config/nturgbd-cross-subject/joint_autodl.yaml` if needed.

## Training and evaluation

```bash
bash train.sh
bash test.sh /path/to/checkpoint.pt
```

The configuration uses seed 4321, 64 frames, batch size 16, 50 epochs and GPU 0. Adjust the config for your hardware. Training outputs are written to `work_dir/ntu60/xsub/simba_joint/`. Additional arguments passed to `train.sh` are forwarded to `main.py`. For a quick check, run `bash train.sh --num-epoch 1` before starting the full experiment.

`bash test.sh <checkpoint> [config]` takes an optional config because the config selects the SSM
operator; `main.py` refuses a checkpoint whose recorded `scan_mode` does not match the config.

## Negative-eigenvalue scan (experimental)

`model_args.scan_mode` selects the Mamba state-retention term:

| `scan_mode` | transition `a[t]` | range |
| --- | --- | --- |
| `baseline` (default) | `exp(delta[t] * A)` | (0, 1) |
| `negative` | `2 * exp(delta[t] * A) - 1` | (-1, 1) |

This follows Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*
(ICLR 2025). It adds no parameters and changes no tensor shape; only the transition inside the
selective scan differs.

The compiled extension must be rebuilt after pulling, because the negative branch lives in
`csrc/selective_scan`:

```bash
rm -rf build && python setup.py build_ext --inplace
python -m unittest tests.test_negative_scan -v
python tools/check_negative_scan_cuda.py     # operator vs reference, forward and all gradients
```

Then, per group:

```bash
bash run_negative.sh negative --num-epoch 10
bash run_negative.sh baseline --num-epoch 10
python tools/scan_statistics.py --config config/nturgbd-cross-subject/joint_autodl_negative.yaml
```

See [docs/negative_eigenvalues_10ep.md](docs/negative_eigenvalues_10ep.md) for the design, the
self-check gate and the pre-registered decision rule.


## Acknowledgements and citation

Simba builds on [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer), [Mamba](https://github.com/state-spaces/mamba) and [Shift-GCN](https://github.com/kchengiva/Shift-GCN).

```bibtex
@article{chaudhuri2024simba,
  title={Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos},
  author={Chaudhuri, Soumyabrata and Bhattacharya, Saumik},
  journal={arXiv preprint arXiv:2404.07645},
  year={2024}
}
```
