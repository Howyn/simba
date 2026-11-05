# 方案 1：Simba + Vim v2 双向时序 SSM

## 来源与接入

直接依据：**Vision Mamba: Efficient Visual Representation Learning with Bidirectional
State Space Model**，2024，ICML 正式接收论文。
[会议论文](https://proceedings.mlr.press/v235/zhu24f.html)；
[作者 v2 核心实现](https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py)。
采用 `bimamba_type=v2` 与 `if_divide_out=true` 的机制，不引入 Vim 的图像位置编码、分类 token 或其它模块。

在现有十个 MambaBlock 的 Mamba 核心处替换，形状始终为 `[N*M,T,V*C]`，NTU 为 `[N*M,T,625]`。
只翻转时间 T，V 仍合并在特征维中。前向与反向共享输入/输出投影，但分别拥有
卷积、x/dt 投影、A、D 参数。反向输出按时间翻转回来，再与正向输出取平均。
外层仍为 `Dropout(RMSNorm(mixer(x) + x))`；GCN、ShiftTCN、降采样位置与池化不改。

代码位于 `mamba_ssm/modules/bimamba.py`，由 `mamba_blocks.py` 的 `use_bimamba` 开关选择。
关闭时创建原 Mamba，原参数名和 forward 保持不变。`use_mti` 与 `use_bimamba` 不允许同时启用。

## 与作者代码的适配差异

- 复用已编译的原版 `mamba_inner_fn`，不安装 Vim 的另一套 CUDA 扩展。
- 作者先融合隐藏状态、再做一次输出线性投影；本实现共享同一输出投影，
  对两方向分别投影后取平均。数学上利用线性性等价，浮点舍入顺序可能不同，
  成本是多一次输出投影计算，不声称逐行复现作者内核。
- 反向参数使用基线 Mamba 的标准初始化，包括 dt 范围 0.001–0.1，
  而非作者 v2 反向 `dt_proj_b` 的普通 Linear 默认初始化。
  这保留基线的步长初始化约定，属于明确的适配选择，不另提出新的机制。
- 额外初始化在隔离 RNG 上下文中完成，保持后续公共权重和数据 RNG 流不变。
- 每块增加 168,750 参数，十块共增加 1,687,500；共享大投影，未直接堆两个完整 Mamba。
- 双向增加计算与激活，不能把基线 11.12 分钟/轮的短测结果当作改进版耗时。

## 两组完全共用的短协议

详见 `EXPERIMENT_PROTOCOL.md`。本次配置为
`config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml` 与
`config/nturgbd-cross-subject/joint_bimamba_enabled10.yaml`。
除工作目录和 `use_bimamba` 外一致；MTI 均关闭。

| 项目 | 数值 |
|---|---|
| 数据 | NTU60 X-Sub，joint，64 帧 |
| 默认种子 | 4321 |
| 轮数 | 10，不自动续至 15/25 |
| 学习率 | 第1轮 0.0125，第2–6轮 0.025，第7–10轮 0.0025 |
| 优化器 | SGD / momentum 0.9 / Nesterov / weight decay 0.0004 |
| batch | 训练24，测试64 |
| 精度 | BF16 autocast + GradScaler，TF32开启；评估FP32 |
| 加载 | 8 workers / pin_memory / prefetch=2 / 不启用persistent workers |
| 观察窗口 | 8–10；±0.2个百分点规则，不稳定不输出稳定均值 |

主程序通过显式 `short10` 分支逐轮计算日程；第7轮自动降档。
恢复训练沿用 checkpoint 的 epoch，不重新预热。
旧 fixed15 协议、旧签名兼容逻辑保留，旧配置未被重写。

## AutoDL 后台启动

```bash
cd /root/autodl-tmp/Simba-git
git pull --ff-only
mkdir -p logs

nohup "$PWD/.venv/bin/python" -u tools/run_bimamba_comparison.py \
  --data-path /root/autodl-tmp/ntu60/ntu/NTU60_CS.npz \
  --output-root work_dir/ntu60/xsub/bimamba_short10_v1 \
  > logs/bimamba10_comparison.log 2>&1 < /dev/null &
echo "训练队列进程：$!"
```

先自动进行小规模融合/参考梯度检查、公共权重检查和完整模型 batch24 的一次合成数据前后向检查，
通过后正式跑基线10轮，再跑BiMamba10轮。检查进程退出后显存释放，不保存检查时的临时权重。
任一检查或基线出错都会停止，不自动降低某一组 batch 或学习率。
关闭SSH或本地断网不影响后台队列；实例需保持运行。

## 查看进度和结果

```bash
tail -n 30 -F logs/bimamba10_comparison.log
```

简洁查看（Ctrl+C退出日志查看不会停止后台训练）：

```bash
grep -aE 'PREFLIGHT|START baseline|START bimamba|EPOCH_RESULT|SCREENING COMPLETE|COMPARISON RESULT|BOTH 10|EXPERIMENT FAILED|Traceback' \
  logs/bimamba10_comparison.log | tail -n 25
```

每轮日志包含 Top-1、实际学习率、训练/评估耗时、当前变体的 ETA。该 ETA 不包含尚未开始的第二组。
基线/改进版的 `log.txt`、`metrics.csv`、`results.json` 以及 latest/best/final checkpoint
分别位于输出根目录的 `baseline_seed4321`、`bimamba_seed4321`。
基线完成后不会覆盖它的结果；第二组从头独立初始化。

完成后：

```bash
cat work_dir/ntu60/xsub/bimamba_short10_v1/comparison.md
```

稳定时输出均值和差值；不稳定时保留逐轮结果与范围，结论为 `inconclusive_unstable`。
稳定且差值大于0时为 `initial_signal`，否则为 `no_signal`。单种子10轮仅作初筛。

## 中断恢复

实例中断后重新启动，先确认旧训练进程已经退出，再执行同一条命令（建议日志改追加 `>>`）。
队列会验证原配置和代码版本，跳过已完成的变体，并自动从未完成变体的 `runs-latest.pt`
恢复模型、优化器、GradScaler、epoch历史和RNG。不会自动增加训练总轮数。
SSH断开但进程仍在时不应重复启动；同一输出根目录有运行锁。

若第一轮结束前失败，没有完整 checkpoint，请使用新的 `--output-root` 重试，避免覆盖旧诊断文件。
若仅单独运行一个变体，可加 `--variant baseline` 或 `--variant bimamba`；默认按两组顺序运行。

本地未具备 AutoDL CUDA 环境时只验证协议/结构；实际 GPU 检查由启动命令在云端执行。
