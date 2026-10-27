# Simba <img src="assets/blue_lion.png" alt="Lion Logo" width="40" height="40">
This is the official implementation of our paper [Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos](https://arxiv.org/abs/2404.07645). To the best of our knowledge, this is the first skeleton action recognition model based on Mamba augmentation with GCN. 

## News

- [ ] Make pre-trained models available.
- [ ] This repo borrows inspiration from multiple codebases. Therefore, some files are redundant. Working towards cleaning the repo !
- [x] Make the code available.

## Simba architecture
<p align="center">
   <img src="model_final_3.png" alt="drawing" width="800"/>
</p>

# Preparation
### Install torchlight
Run `pip install -e torchlight`
### Install dependencies
```
pip install -r requirements.txt
```

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Evaluation

```
bash evaluate.sh
```
Replace path_to_model_weights argument to the saved model weights you obtain after training.
# Training & Testing

### Training

```
bash train.sh
```

Please check the configuration in the config directory.

### Testing

```
bash evaluate.sh
```

To ensemble the results of different modalities, run the following command:

```
bash ensemble.sh
```

## Acknowledgements

This repo is based on [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer), [Mamba](https://github.com/state-spaces/mamba) and [Shift-GCN](https://github.com/kchengiva/Shift-GCN).

Thanks to the original authors for their work!

## Citation

Please cite this work if you find it useful:
````BibTeX
@article{chaudhuri2024simba,
  title={Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos},
  author={Chaudhuri, Soumyabrata and Bhattacharya, Saumik},
  journal={arXiv preprint arXiv:2404.07645},
  year={2024}
}
````

# Contact
For any questions, feel free to raise an issue !

## AutoDL: NTU RGB+D 60 X-Sub joint

The repository includes a single-GPU configuration at
`config/nturgbd-cross-subject/joint_autodl.yaml`. It reads the processed
dataset from `/root/autodl-fs/ntu/NTU60_CS.npz`, so the raw archive and the
large generated data file stay outside the Git repository.

On an Ubuntu AutoDL instance, run:

```bash
cd /root/autodl-tmp
git clone git@github.com:Howyn/simba.git Simba-main
cd Simba-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-autodl.txt
python -m pip install -e . --no-build-isolation
python -m pip install -e torchlight
```

Extract the NTU archive outside the project and generate the NPZ file:

```bash
mkdir -p /root/autodl-fs/nturgbd_raw
unzip /root/autodl-tmp/nturgbd_skeletons_s001_to_s017.zip -d /root/autodl-fs/nturgbd_raw
python tools/prepare_ntu60_xsub.py
```

The converter prints the train and test shapes. They should be compatible
with `feeders.feeder_ntu.Feeder` and the output should exist at
`/root/autodl-fs/ntu/NTU60_CS.npz`.

Run a one-epoch smoke test first by temporarily changing `num_epoch` to `1`
in the AutoDL config, then restore it to `50` for the experiment:

```bash
screen -S simba
bash train.sh
# Ctrl-A, D to detach; reattach later with: screen -r simba
```

The RTX 4090 uses GPU `0`. Checkpoints and logs are written under
`work_dir/ntu60/xsub/simba_joint/`. To evaluate a checkpoint:

```bash
bash test.sh work_dir/ntu60/xsub/simba_joint/runs-50-*.pt
```

