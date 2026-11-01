# Simba 内部 SSM 的 20 epoch 四组对照

本轮固定 NTU RGB+D 60、X-Sub、Joint，四组均从头训练 20 epoch。结果写入新的 `full_*_20ep` 目录；目录中的 full 只是这一批实验的标签，20 epoch 仍是短预算实验，不能等同于完整收敛。保留已跑过的 10 epoch 文件和目录。

## 前四轮的信号

| Epoch | 旧基线 Top1 | 组合组 Top1 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| 1 | 42.16% | 41.78% | -0.38 |
| 2 | 56.06% | 55.29% | -0.77 |
| 3 | 60.16% | 64.13% | +3.97 |
| 4 | 65.82% | 67.76% | +1.94 |

第 3、4 轮出现积极信号，但仍处于前五轮 warmup，不能据此认定最终会稳定提升约两个点。旧基线截图中的 10 轮最高 Top1 为 73.19%（第 9 轮），第 10 轮为 73.07%；当前组合组尚未跑完相同预算。旧基线据记录在 4090 上运行，当前日志是 3080 Ti，本轮四组应在同一 GPU 和环境下重跑。

## 20 epoch 能不能达到论文的 89%

不能保证。依据 [Simba 原论文第 4.2 节、表 2 与附录表 7](https://arxiv.org/html/2404.07645v1)，NTU 采用 90 epoch，学习率在第 75、85 个 epoch 衰减；NTU60 X-Sub 单 Joint Top1 为 89.03%。本轮采用 20 epoch 是计算预算选择，并非论文指标的复现承诺。

论文训练 batch 为 64、测试 batch 为 512，Mamba embedding 为 500；当前工作代码使用训练/测试 batch 16、embedding 625。即便把轮数增加到论文预算，也不能只凭 epoch 数保证相同结果。当前先在固定 20 epoch 下判断相对本地基线的收益，若有效再决定是否延长。

## 四组配置

| 组别 | 模式 | 双向扫描层 | 运动 Δ 层 | 目录（均位于 `work_dir/ntu60/xsub/`） |
| --- | --- | --- | --- | --- |
| Baseline | `baseline` | 无 | 无 | `full_baseline_20ep` |
| A | `ssm_bidir` | 4、7、10 | 无 | `full_ssm_bidir_20ep` |
| B | `ssm_motion` | 无 | 4、7、10 | `full_ssm_motion_20ep` |
| A+B | `ssm_bidir_motion` | 4、7、10 | 4、7、10 | `full_ssm_bidir_motion_20ep` |

配置文件为 `config/nturgbd-cross-subject/joint_<mode>_20ep.yaml`。四组都关闭 CTR-GC 和 FR-Head。方向门初始反向占比 0.1；运动支路秩 16、修正尺度 0.1、末层零初始化。

四份配置除模块开关和输出目录外，采用相同设置：

- 从头训练 20 epoch；SGD、momentum 0.9、Nesterov，学习率 0.025，weight decay 0.0004。
- Warmup 5 epoch，`lr_type: multistep`，`step: [14, 18]`，每次衰减十倍。当前 epoch 索引从 0 开始，因此日志第 15、19 轮开始分别使用 0.0025、0.00025。
- 训练/测试 batch 16，64 帧，seed 4321，GPU 0，FP32，显式关闭 TF32。
- Worker 8、prefetch 16、persistent workers 关闭，保留原数据划分和增强。
- NPZ 路径为 `/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz`。

旧日志中的 `lr_type: cosine` 只是标签：原程序每轮实际调用手动 warmup + step 函数，cosine scheduler 的 step 被注释。本轮显式设置 multistep，且将两次衰减安排在 20 epoch 预算内。不要只对某一组单独改变 batch、精度、学习率或 seed。

## 停止当前 10 epoch 队列，再同步

先在正在运行旧三组 `&&` 命令的原终端按一次 `Ctrl+C`。当前前台训练被中断后退出码非零，后面两组不会自动继续。确认旧训练已退出：

```bash
nvidia-smi
pgrep -af 'python.*main.py'
```

新四组从头开始，不带旧实验的 `--weights` 或 `--resume`，也不把已经跑过的四轮计入新的 20 epoch。

在现有云端仓库、原 Python 环境同步代码：

```bash
cd /root/autodl-tmp/Simba-git
source .venv/bin/activate
source /etc/network_turbo
git pull --ff-only origin main
```

本次更新涉及训练检查点、后台管理脚本与配置，云端必须同步完整版本。Git 若提示本地修改冲突或网络失败，先处理同步再启动；不能只恢复一个入口脚本。无需重新编译 CUDA 扩展。

## 后台启动

同一张 GPU 顺序运行四组，默认顺序为 Baseline、A、B、A+B：

```bash
bash run_full_experiments.sh start
```

若希望先花较少预算判断主组合，可以先运行 Baseline 和 A+B：

```bash
bash run_full_experiments.sh start --modes baseline ssm_bidir_motion
```

等这两组都结束，再补两组单模块消融：

```bash
bash run_full_experiments.sh start --modes ssm_bidir ssm_motion
```

后台进程独立于 SSH 会话，因此 SSH 断开或本地电脑断网不会使训练停止。云实例关机、回收或训练进程报错仍会中断；这类中断需要根据检查点恢复。不要在旧训练尚未退出时另启新队列。

## 随时查看

查看队列与当前实验：

```bash
bash run_full_experiments.sh status
```

查看实时输出，默认跟随当前组并切换到下一组：

```bash
bash run_full_experiments.sh logs --follow
```

只跟踪组合组：

```bash
bash run_full_experiments.sh logs --mode ssm_bidir_motion --follow
```

也可直接查看某一组的模型日志：

```bash
tail -n 40 -f work_dir/ntu60/xsub/full_ssm_bidir_motion_20ep/log.txt
```

查看结果汇总：

```bash
bash run_full_experiments.sh results
```

或直接提取每组结束时的最高 Top1、最佳轮数：

```bash
grep -H -E 'Best accuracy:|Epoch number:|Model name:' \
  work_dir/ntu60/xsub/full_*_20ep/log.txt
```

`Best accuracy` 使用 0～1 比例，乘以 100 得到百分数；`Epoch number` 是最高 Top1 对应的轮数。每轮日志中的 `Top1`、`Top5` 已是百分数。

## 权重与比较

各组目录保存 `runs-best.pt`（最高 Top1）、`runs-latest.pt`（最近完整 epoch，每轮覆盖）、第 10/20 轮周期检查点、`log.txt` 和实际 `config.yaml`。

实例重启或报错解决后，重新执行相同的 `start --modes ...` 命令：已完成组会跳过，未完成组从 `runs-latest.pt` 恢复，训练到总计 20 轮。恢复保留模型、优化器和最佳指标；随机增强序列不保证与完全不中断时逐位一致。若第一轮尚未生成检查点就中断，脚本会保留现场并拒绝自动覆盖，先查看对应 `console.log`。`logs --follow` 中按 Ctrl+C 只退出查看，不会停止后台训练。

比较四组最高 Top1、对应 Top5、最佳轮数及第 20 轮 Top1。增益采用百分点，即改进组 Top1 减本轮 Baseline Top1；不要将本地基线替换成论文的 89.03%。这是一个 seed 的短预算消融，若收益很小，需要复核随机波动；若明显有效，再决定是否延长相同预算的基线与改进版。

按当前组合组日志，每轮训练约 658 秒、评估约 67 秒，20 轮粗估约 4 小时，四组约 16 小时；基线可能略快，实际耗时取决于 GPU 与数据负载。

## 收益很小时先检查已有结果

先不要自动追加两组训练或增加 epoch。训练结束时已生成两组最佳模型的 `runs-best_right.txt`，其中实际包含所有测试样本的预测与真值。以下命令仅用 CPU 读取这些文件及组合组检查点，不加载整网或 NPZ，不训练、不运行 GPU 推理，也不覆盖现有结果：

```bash
python tools/diagnose_ssm_results.py
```

输出配对样本数、修正的错误与新增的错误、类别差异，以及 l4/l7/l10 实际方向门和运动分支参数统计。`--skip-checkpoint` 可只分析文本预测。配对要求同一测试集与顺序，权重变化不等于实际有效贡献；方向门变化也可能包含 weight decay 的作用。单次配对结果不能替代多 seed 稳定性验证，也不能判断独立训练的两个单模块是否有效。

## 检查新增分支是否被权重衰减支配

原训练器把所有参数放入同一个 SGD 衰减组。按本轮 2520 批次/轮、warmup 5 轮、学习率 0.025、step `[14,18]`、momentum 0.9、Nesterov 和 weight decay 0.0004，模拟零任务梯度的 19 轮更新，初始方向门 0.1 会变成约 0.475995。普通线性权重会收缩到初始值的约 0.04373 倍；1250 输入的运动下投影初始 RMS 理论值约 0.01633，纯衰减后约 0.000714。若实测吻合，门值变化不能解释为学到了反向依赖；这也不证明所有任务梯度始终为零。

以下短检查只读取已有组合权重，取两个小批次计算实际 Δ 改变量及任务梯度/原衰减项比例，不创建优化器、不更新参数、不写训练结果：

```bash
python tools/probe_ssm_signal.py --batches 2 --batch-size 2
```

程序使用 GPU 前向和反向计算，含数据初始化耗时；它不是两个 epoch。内存中的 BatchNorm 统计可随 train 模式更新，但不会保存。小批次梯度只是局部诊断，不能作为最终涨点的证据。

训练器新增显式开关 `--ssm-protect-addons True`，只把新增 `bidir_logit`、`motion_down.weight`、`motion_up.weight` 放入零 weight decay 组。默认关闭，旧配置和已完成结果保持原样；主干、原 Mamba 参数与其衰减策略均不改。该修正用于排除新增分支的衰减干扰，本身不是新增算法贡献，也不保证涨点。

不要直接在原目录恢复旧单组优化器并启用此开关：保护后的优化器有两个参数组，程序会明确拒绝不匹配的恢复。先查看上述信号检查结果，再决定是否在新目录从头训练修正版组合；当前不自动追加 20 轮，也不改成全局关闭主干衰减，以免丢失原基线的可比性。
