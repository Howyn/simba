# Simba 内部 SSM 的 10 epoch 筛选

本轮只比较内部双向状态扫描和运动条件化 Δ。三份新配置均关闭 CTR-GC 和 FR-Head，保留原始空间主干与交叉熵损失。新增模块统一放在 l4、l7、l10，覆盖 64、32、16 帧的三个阶段。

| 组别 | 命令模式 | 双向状态扫描 | 运动条件化 Δ | 默认结果目录 |
| --- | --- | --- | --- | --- |
| 已有基线 | baseline | 无 | 无 | work_dir/ntu60/xsub/experiment_baseline_10ep |
| A | ssm_bidir | l4、l7、l10 | 无 | work_dir/ntu60/xsub/experiment_ssm_bidir_10ep |
| B | ssm_motion | 无 | l4、l7、l10 | work_dir/ntu60/xsub/experiment_ssm_motion_10ep |
| A+B | ssm_bidir_motion | l4、l7、l10 | l4、l7、l10 | work_dir/ntu60/xsub/experiment_ssm_bidir_motion_10ep |

配置位于 `config/nturgbd-cross-subject/joint_ssm_*_10ep.yaml`。数据集为 NTU RGB+D 60、X-Sub、Joint；64 帧，seed 4321，batch size 16，单卡 GPU 0，SGD，学习率 0.025，weight decay 0.0004，warmup 5 epoch，学习率衰减点 40，总计 10 epoch。这些训练设置与仓库中的 `joint_baseline_10ep.yaml` 相同。

## 算法范围

A 只执行一次输入投影、局部卷积与 Δ/B/C 生成，然后共享这些编码进行正、反两次状态扫描；反向输出恢复原时间顺序后融合。D 跳连、z 输出门控和输出投影各执行一次。方向融合门初始化为偏向原正向扫描，反向占比为 0.1。它是共享局部编码的双向状态扫描，不是复制两套完整 Mamba。参数增量较小，但扫描计算、显存和重排开销会增加。

B 将相邻时刻内部特征的差分送入低秩支路，在原有 Δ 的低秩表示中增加有界的小幅修正，再沿用原 `dt_proj` 和 softplus。运动支路秩为 16，修正尺度为 0.1。第一版不改 A、B、C 和状态维数。首次训练从接近原模型的 Δ 开始，随后学习运动条件修正。

以上为适配当前 Simba 代码的实现，实际增益由下面的消融实验验证。

## 复用已跑完的基线

原基线配置和默认模型路径保留。已有 10 epoch 指标可以继续使用，无需仅因为增加可选模块而重跑。

比较时检查旧实验保存的参数、日志与本轮设置一致：同一 NPZ、同一 4090 环境、相同 batch size、帧数、seed、优化器、精度、TF32、worker 和数据加载选项。尤其不要把旧 FP32 基线与新 BF16 或不同 TF32 设置直接归因于模块差异。

以下训练命令沿用训练程序原有默认值，不自动开启 BF16、TF32，也不改 batch size。若旧基线通过命令行另设了这些参数，三组新实验应追加完全相同的选项。

## 在 AutoDL 拉取代码

在现有 Simba 环境中执行；仓库目录或虚拟环境名称不同则替换为实际路径。本次只修改 Python 与配置，无需重新安装环境或编译 CUDA 扩展。

```bash
cd /root/autodl-tmp/Simba-main
source .venv/bin/activate
git pull --ff-only origin main
git rev-parse --short HEAD
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

如果 `git pull --ff-only` 提示云端工作区有冲突或分支已经分叉，保留本地结果和修改后再处理，不使用强制覆盖命令。

## 先做 GPU 自检

先校验原生 CUDA scan 与参考递推的输出、梯度，以及零初始化运动分支的基线一致性：

```bash
python tools/check_ssm_cuda.py
```

每个模式使用一批随机骨架，检查前向、反向、优化器更新、评估输出、新增模块位置与梯度。它们不读取数据集，不写训练结果。

```bash
python tools/smoke_academic_model.py ssm_bidir
python tools/smoke_academic_model.py ssm_motion
python tools/smoke_academic_model.py ssm_bidir_motion
```

这些命令通过，说明当前 CUDA 环境能够执行新路径，不代表已经取得分类涨点。若旧实验采用 BF16，可以在所有自检命令中同样追加 `--precision bf16`。原生 scan 数值检查固定关闭 TF32；整网 smoke 可用 `--tf32` 开关检查启用 TF32 的运行情况，正式训练仍应与旧基线保持一致。

## 三组新实验

使用已有基线作为第四组对照，新增三组均从头训练，不带 `--weights` 或 `--resume`。载入旧基线权重会变成额外预训练后微调，不能直接与从头训练的 10 epoch 基线比较。

在同一张 4090 上依次运行：

```bash
bash run_experiment.sh ssm_bidir
bash run_experiment.sh ssm_motion
bash run_experiment.sh ssm_bidir_motion
```

每条命令都训练 10 epoch，结果目录互不覆盖，也不会写入已有基线目录。可在同一个 `screen` 会话中依次执行；不要并行占用同一张 4090。

脚本检测到目标目录已经存在 `log.txt` 时会停止。需要复跑时换目录，例如：

```bash
bash run_experiment.sh ssm_bidir_motion \
  --work-dir work_dir/ntu60/xsub/experiment_ssm_bidir_motion_10ep_repeat2
```

断点恢复使用 `main.py`，并指定同一配置、原目录和对应检查点；它与新的一次独立训练不同。

## 查看与比较结果

每组记录相同口径的最高 Top1、对应 Top5、最佳 epoch、第 10 epoch Top1，以及训练耗时和峰值显存。优先查看 A、B 是否分别优于原基线，再判断 A+B 是否具有互补收益。三个模型参数和优化难度有差异，不能仅用某一个 epoch 的波动证明最终收益。

最高 Top1 对应的检查点为各组目录内的 `runs-best.pt`。单独评估组合组：

```bash
python main.py \
  --config config/nturgbd-cross-subject/joint_ssm_bidir_motion_10ep.yaml \
  --phase test \
  --weights work_dir/ntu60/xsub/experiment_ssm_bidir_motion_10ep/runs-best.pt \
  --work-dir work_dir/ntu60/xsub/eval_ssm_bidir_motion_10ep \
  --save-score True
```

10 epoch 用于初步筛选，尚未经过第 40 epoch 的学习率衰减。有效候选进入完整训练时，应为原基线和改进版采用一致的训练预算与策略，并创建新的结果目录；不要把论文原始完整训练指标当作本地 10 epoch 实验的起点。

## 可选的 4090 吞吐检查

以下命令使用真实训练数据做短时测量，不保存检查点或实验日志。不是正式训练的必要步骤；只在需要判断计算开销或显存时使用。

```bash
python tools/benchmark_4090.py ssm_bidir --precision fp32 --tf32 false
python tools/benchmark_4090.py ssm_motion --precision fp32 --tf32 false
python tools/benchmark_4090.py ssm_bidir_motion --precision fp32 --tf32 false
```

基线和改进版应在同一精度设置下测量。不要为了让组合组跑得更快而只修改它的 batch size、精度或帧数。
