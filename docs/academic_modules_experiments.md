# Simba 模块实验

本目录之外的原始 `joint_autodl.yaml` 和 `train.sh` 继续作为原基线使用。以下四组均采用 NTU RGB+D 60、X-Sub、Joint、单卡、10 epoch 和相同的数据路径及训练参数，只改变模块开关。10 epoch 主要用于筛选，不能单独证明最终收敛精度。

| 组别 | 配置 | 入口空间模块 | 辅助头 |
| --- | --- | --- | --- |
| baseline | `joint_baseline_10ep.yaml` | 原始 Shift-GCN | 无 |
| ctr | `joint_ctr_10ep.yaml` | l8、l9、l10 换为 CTR-GC | 无 |
| fr | `joint_fr_10ep.yaml` | 原始 Shift-GCN | l5、l8、l10 |
| ctr_fr | `joint_ctr_fr_10ep.yaml` | l8、l9、l10 换为 CTR-GC | l5、l8、l10 |

配置位于 `config/nturgbd-cross-subject/`。上表中的 CTR 指选中 Simba 块入口的空间单元替换，不替换整个主干。FR 采用时空分支、类别原型与 FN/FP 校准；这是适配 Simba 的实现，选用官方 V0 形式和三层权重 `[0.2, 0.5, 1.0]`，并非完整复刻官方四层实验。训练损失为 `CE + 0.1 * FR`。FR 原型是模型缓冲区，随检查点保存。推理阶段不计算辅助头。

参考：

- CTR-GCN：Chen et al., ICCV 2021，https://github.com/Uason-Chen/CTR-GCN 。其代码许可为 CC BY-NC 4.0，使用时请遵循上游许可。
- FR-Head：Zhou et al., CVPR 2023，https://github.com/zhysora/FR-Head 。

## 开始前

先把代码同步至 AutoDL，进入原 Simba 虚拟环境，确认数据文件 `/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz` 与基线所用的是同一份。四组配置默认输出到 `work_dir/ntu60/xsub/experiment_<组别>_10ep/`，不会写入原 `baseline_10ep`。若修改了骨架帧数，FR 的 `model_args.fr_input_frames` 应与 feeder 的 `window_size` 一致（默认均为 64）。

先检查两个模块的前向、反向和状态保存，再检查完整 Simba 的一批合成数据：

```bash
python -m unittest discover -s tests -p test_academic_modules.py
python tools/smoke_academic_model.py ctr_fr
```

自检不读取数据集，不代表 10 epoch 训练已经成功。显存不足时记录错误；不要只降低其中某一组的 batch size 后直接比较。

## 四组训练

```bash
bash run_experiment.sh baseline
bash run_experiment.sh ctr
bash run_experiment.sh fr
bash run_experiment.sh ctr_fr
```

脚本发现目标目录已有 `log.txt` 时会停止，避免把两次实验的日志混在一起。复跑时指定新的目录，例如：

```bash
bash run_experiment.sh ctr_fr --work-dir work_dir/ntu60/xsub/experiment_ctr_fr_seed4321_repeat2
```

`runs-best.pt` 是评估 Top1 最高轮次的检查点，包含模型、FR 原型和优化器状态；周期检查点及最终轮次检查点也会保留。训练日志中的 `loss2` 是已经乘过 `fr_loss_weight` 的辅助损失。四组应分别记录 Top1、Top5、最佳轮次、训练时间与显存；若需要评估混淆类别，保留自动生成的每类准确率和混淆矩阵文件。

使用与权重对应的配置进行单独评估，例如：

```bash
python main.py --config config/nturgbd-cross-subject/joint_ctr_fr_10ep.yaml \
  --phase test --weights work_dir/ntu60/xsub/experiment_ctr_fr_10ep/runs-best.pt \
  --work-dir work_dir/ntu60/xsub/eval_ctr_fr_10ep --save-score True
```

如果从训练检查点恢复，使用同一组配置和原工作目录调用 `python main.py --config ... --work-dir ... --resume ...`。若需要继续增加总轮数，应通过 `--num-epoch` 显式设置新的总轮数。FR 原型更新当前只支持单 GPU；暂不支持 FR 与模型 EMA 同时启用。

## 4090 训练速度

训练循环合并了逐 batch 的指标同步，使用固定内存和异步数据传输。默认仍使用 FP32，保持原有 batch size、数据处理和训练轮数。每个 epoch 的日志会显示训练耗时、每秒处理的样本数及峰值已分配显存，耗时包括数据加载。

可选使用 BF16 混合精度和 TF32：

```bash
python tools/smoke_academic_model.py ctr_fr --precision bf16 --tf32
bash run_experiment.sh ctr_fr --precision bf16 --tf32 true --persistent-workers true --prefetch-factor 2
```

先在同一台 GPU 上用真实训练数据测量短时吞吐；每次测量都创建独立的临时模型，不写训练权重或实验日志：

```bash
python tools/benchmark_4090.py ctr_fr --precision fp32 --tf32 false
python tools/benchmark_4090.py ctr_fr --precision bf16 --tf32 true
```

两次测量都使用相同配置的 batch size，不读取测试集，输出热身后的样本/秒和峰值显存。测量结果受其他任务、数据加载和 GPU 负载影响；建议先确保 GPU 空闲并多测一次。若 BF16 无法运行、出现非有限数值或实际更慢，改用 `--precision fp32 --tf32 false` 并保留数据传输、日志同步优化。评估使用 FP32 张量，但如果开启 `--tf32 true`，部分矩阵乘法和卷积的内部计算精度仍可能变化。

启用 BF16、TF32 或持久化数据加载进程可能改变数值或随机数据增强序列。正式消融时，四组应使用同样的精度、TF32、worker、batch size 和数据设置，并在 4090 上重新跑基线。暂不自动增大 batch size：较大 batch 会改变 BN、FR 原型统计及优化步数，需要单独实验。多 worker 时 `--prefetch-factor` 控制每个 worker 预取的 batch 数；大数据集和较多 worker 会增加主机内存占用，视服务器实际资源调整。
