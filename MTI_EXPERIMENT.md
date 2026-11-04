# Simba + MTI：NTU60 X-Sub joint 初筛

仅引入方案 2 的 MTI。基线参考提交为 `94478ab`，空间 GCN、ShiftTCN、
Mamba 单向扫描、状态维度和全局池化均保留。没有加入双向或逐关节扫描。

## 方法来源与适配约定

- **TSkel-Mamba: Temporal Dynamic Modeling via State Space Model for Human
  Skeleton-based Action Recognition**，2025，arXiv:2512.11503（预印本），已接收
  BMVC 2026。依据：[论文](https://arxiv.org/html/2512.11503v1)、
  [会议日程](https://bmvc2026.bmva.org/programme/wednesday/)。
- 使用公式 (4) 的并行残差聚合：`H + Cycle1(H) + Cycle3(H) + Cycle5(H)`。
  算法 1 的伪代码写成了顺序更新；本实现明确选择公式 (4)。
- 公式 (3) 将偏移写成 `c % K - 1`，仅与正文 K=3 的中心对称示例一致。
  这里明确采用该示例的中心对称多尺度扩展 `c % K - K//2`：K=1 为 0，
  K=3 为 -1/0/1，K=5 为 -2/-1/0/1/2；时间越界补零。
  Cycle FC 的固定偏移/投影机制来自论文引用的较早基础工作 CycleMLP，
  不是本项目提出的新机制；可对照其
  [作者实现](https://github.com/ShoufaChen/CycleMLP/blob/main/cycle_mlp.py)。
  这些约定是适配选择，不声称逐行复现尚未发布的 TSkel-Mamba 作者代码。
- 在每个 MambaBlock 内，把 `[B,T,625]` 恢复为 `[B,25,T,25]`，每个关节分别
  做时间/通道交互，再展平送入原 Mamba。块外残差仍使用原输入。
- 十个块全部使用同一模块、同一尺度。每块新增 1,950 个参数，合计 19,500。
- `model_args.use_mti: false` 不构造 MTI，不改变原参数名和 forward 表达式。
  MTI 额外初始化使用独立的 RNG 上下文，不消耗后续基线权重的初始化随机数。

## 两组共同配置

| 设置 | 数值 |
|---|---|
| 数据 | NTU60 / X-Sub / joint，64 帧，同一 NPZ |
| 默认数据路径 | `/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz` |
| 随机种子 | 4321 |
| 优化器 | SGD，momentum=0.9，Nesterov=true |
| 权重衰减 | 0.0004 |
| 学习率 | **0.005，从 epoch 0 起固定**，无预热、无衰减 |
| batch size | 训练 16，评估 16 |
| 精度 | 训练 BF16 autocast + GradScaler（初始 scale=1）；评估 FP32 |
| 数据加载 | workers=8，pin_memory=true，prefetch=2，persistent_workers=false |
| 时长 | 15 epoch，每个 epoch 评估并保存 latest |
| 显卡 | 单卡，进程内 GPU 0；不使用 DDP |

0.005 是本轮初筛选定的学习率，不是已经证实的最优值，也不保证比 0.025
更快收敛。两组共用它；本轮不允许只给一组调参。
最初 FP16 配置在基线第一轮出现非有限训练指标，因此两组统一改为 BF16 AMP，
保留 GradScaler，初始 scale=1。此项调整用于规避 FP16 的有限数值范围，尚需
在实际显卡上验证。失败的 FP16 运行不参与比较，也不从其权重恢复；重新从头
运行两组，并用新的 `--output-root`（例如 `work_dir/ntu60/xsub/mti_bf16_v2`）。
若 BF16 仍出现非有限损失，日志会报告具体 batch 及输入/输出是否有限，供继续定位。
两个 YAML 除 `work_dir` 和 `model_args.use_mti` 外完全相同；原 `joint_autodl.yaml`
保留，不要用它启动本轮初筛。

## AutoDL：先只启动基线

先更新代码；若 Git 报本地修改冲突，先处理自己的修改，不要继续启动旧代码：

```bash
cd /root/autodl-tmp/Simba-git
git pull --ff-only
```

使用此前已经能运行基线的 `.venv`。MTI 没有新增第三方依赖或 CUDA 扩展。
如果原版 selective scan 的本地扩展尚未编译，执行
`MAX_JOBS=2 .venv/bin/python setup_baseline_scan.py build_ext --inplace`；已有匹配的扩展
无需重新编译。必须使用同一个虚拟环境，避免加载全局的不同接口版本。

可选数值检查（只做小张量计算，不跑数据集训练）：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_mti*.py' -v
```

启动基线，关闭本地网络或 SSH 不会停止训练：

```bash
mkdir -p logs
nohup "$PWD/.venv/bin/python" -u tools/run_mti_experiment.py --variant baseline \
  > logs/mti15_baseline_seed4321.log 2>&1 < /dev/null &
echo "launcher PID: $!"
```

启动器直接使用当前 Python 的绝对路径启动训练，不依赖后台 shell 激活环境。
它只运行指定的一组，不会自动接着启动 MTI。
`nohup` 保护 SSH 断开；实例关机/回收仍会中断进程，可以从 epoch 边界检查点恢复。

## 查看进度与剩余时间

```bash
grep -aE 'EPOCH_RESULT|AMP enabled|SCREENING COMPLETE|EXPERIMENT FAILED|Traceback' \
  logs/mti15_baseline_seed4321.log | tail -n 20
nvidia-smi
cat work_dir/ntu60/xsub/mti_fixed15/baseline_seed4321/launcher_status.json
```

第一个完整 epoch 结束后，`EPOCH_RESULT` 会包含编号、top-1、LR、耗时和
`ETA_minutes`。ETA 根据已经完成的 epoch 平均耗时估算，包含训练与评估，
不含首次初始化和检查点写盘时间。每轮还记录 samples/second、显存峰值、
GradScaler scale；日志报告前向输出 dtype，以核对 AMP 是否实际开启。
具体吞吐提升需要与相同硬件上的测量比较，代码不预设加速倍数。

## 基线恢复

先确认旧进程已结束，再执行；不要在仍运行时重复启动：

```bash
nohup "$PWD/.venv/bin/python" -u tools/run_mti_experiment.py --variant baseline --resume \
  >> logs/mti15_baseline_seed4321.log 2>&1 < /dev/null &
echo "launcher PID: $!"
```

从 `runs-latest.pt` 恢复模型、优化器、GradScaler、已完成 epoch 记录和全局 RNG。
不从 best 权重或旧实验权重继续训练。恢复时核对模型开关、学习率、精度、种子、
batch size 和数据管道，设置不同会拒绝恢复。默认非持久 DataLoader workers 支持
epoch 边界恢复；CUDA 非确定性算子仍可能造成微小差异。

## 结果与稳定判据

每组输出目录中包含：

- `launch_config.yaml`：启动参数；`config.yaml`：训练解析后的完整参数。
- `log.txt`：训练日志；`metrics.csv`：每 epoch 的 top-1、LR、耗时、吞吐与显存。
- `runs-latest.pt`：最近完成的 epoch；`runs-best.pt`：最佳 epoch。
- `runs-final.pt`：第 15 epoch 最终状态，保留优化器和 Scaler，可审计。
- `results.json`：逐轮记录及第 10–15 epoch 的稳定判定。
- `launcher_status.json`：启动器 PID、源码提交、状态及退出码。

top-1 全部以百分数记录，例如 85.2，而不是 0.852。
六轮数值围绕其均值的最大偏差不超过 **0.2 个百分点**时，`stability.status=stable`，
`final_mean_top1_percent` 为六轮均值。未稳定时该字段为 `null`，保留范围和偏差，
不强行输出最终均值。不足六轮时为 `incomplete`。
当前评估使用官方 test 划分，不是独立验证集；本轮属于探索性初筛。

基线结束后发回 `results.json` 和六轮数据，按原阶段门禁确认后再进入改进版。

## 确认基线后，再启动 MTI

以下是后续步骤的命令，不会由启动器自动执行：

```bash
nohup "$PWD/.venv/bin/python" -u tools/run_mti_experiment.py --variant mti \
  > logs/mti15_mti_seed4321.log 2>&1 < /dev/null &
echo "launcher PID: $!"
```

改进版目录为 `work_dir/ntu60/xsub/mti_fixed15/mti_seed4321`。
恢复时同样添加 `--resume` 并使用 `>>` 追加日志。
额外种子用 `--seed` 指定，两组必须同步补跑；初筛前不自动补种子。
自定义数据位置可传 `--data-path /absolute/path/NTU60_CS.npz`，两组必须一致。
启动器会拒绝覆盖已有实验目录；需要从头重跑时，用新的 `--output-root` 保存。
