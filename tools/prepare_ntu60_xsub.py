"""Convert NTU RGB+D 60 skeleton files to Simba's NPZ format."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
NAME_RE = re.compile(r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<rep>\d{3})A(?P<action>\d{3})\.skeleton$", re.I)


def read_skeleton(path: Path, max_frames: int) -> np.ndarray:
    """Return one file as (T, 2, 25, 3), keeping the two largest bodies."""
    with path.open("r", encoding="utf-8") as f:
        n_frames = min(int(f.readline()), max_frames)
        data = np.zeros((max_frames, 2, 25, 3), dtype=np.float32)
        for frame_idx in range(n_frames):
            n_bodies = int(f.readline())
            bodies = []
            for _ in range(n_bodies):
                f.readline()  # body id and body metadata
                n_joints = int(f.readline())
                joints = np.zeros((25, 3), dtype=np.float32)
                for joint_idx in range(n_joints):
                    values = f.readline().split()
                    if joint_idx < 25:
                        joints[joint_idx] = [float(values[0]), float(values[1]), float(values[2])]
                bodies.append((n_joints, joints))
            bodies.sort(key=lambda item: item[0], reverse=True)
            for person_idx, (_, joints) in enumerate(bodies[:2]):
                data[frame_idx, person_idx] = joints
    return data.reshape(max_frames, 150)


def parse_name(path: Path):
    match = NAME_RE.match(path.name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def convert(raw_dir: Path, output: Path, max_frames: int) -> None:
    files = sorted(raw_dir.rglob("*.skeleton"))
    if not files:
        raise FileNotFoundError(f"No .skeleton files found below {raw_dir}")
    train_files, test_files = [], []
    for path in files:
        info = parse_name(path)
        if info is None or not 1 <= info["action"] <= 60:
            continue
        (train_files if info["subject"] in TRAIN_SUBJECTS else test_files).append((path, info["action"] - 1))

    def build(items):
        x = np.empty((len(items), max_frames, 150), dtype=np.float32)
        y = np.zeros((len(items), 60), dtype=np.float32)
        for index, (path, label) in enumerate(tqdm(items, desc="Converting", unit="file")):
            x[index] = read_skeleton(path, max_frames)
            y[index, label] = 1.0
        return x, y

    x_train, y_train = build(train_files)
    x_test, y_test = build(test_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Wrote {output}")
    print(f"train: {x_train.shape}, test: {x_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/root/autodl-fs/nturgbd_raw"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-fs/ntu/NTU60_CS.npz"))
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    convert(args.raw_dir, args.output, args.max_frames)
