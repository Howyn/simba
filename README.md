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
