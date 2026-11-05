# AutoDL 短测速（3080 Ti）

用途：先花几分钟确定更快的运行设置，再开始正式实验。默认比较以下四组：

| 名称 | 训练 batch | 测试 batch | TF32 |
|---|---:|---:|---|
| reference_b16_eval16 | 16 | 16 | 关闭 |
| b16_eval64 | 16 | 64 | 关闭 |
| b24_eval64 | 24 | 64 | 关闭 |
| b24_eval64_tf32 | 24 | 64 | 开启 |

全部使用真实 NTU 数据、原版 Simba、BF16 AMP + GradScaler、FP32 评估、8 个加载进程。
直接调用当前 `main.py` 的 `Processor.train/eval` 方法，保留原有数值检查和数据增强。
每组先预热 15 个训练 batch，再计时 80 个；评估先预热 5 个 batch，再计时 20 个。
预热和计时共用数据迭代器，计时不重复启动加载进程。每组在独立子进程运行，OOM 会记录并跳过。

这不是正式训练：不读取或覆盖已有 checkpoint，不保存权重，不修改模型或训练配置。
临时更新的模型随子进程退出丢弃。子日志中可能出现准确率，它没有实验意义。
每组会在自己的测速目录写一个临时分类统计 CSV，这是原评估方法的输出。

## 启动

先确保已有训练完成，避免多个任务争抢同一张 GPU。默认检查到 GPU 计算进程会停止测速。

```bash
cd /root/autodl-tmp/Simba-git
git pull --ff-only
mkdir -p logs

nohup "$PWD/.venv/bin/python" -u tools/benchmark_autodl.py \
  --data-path /root/autodl-tmp/ntu60/ntu/NTU60_CS.npz \
  --output-dir work_dir/speed_probe_v1 \
  > logs/speed_probe_v1.log 2>&1 < /dev/null &
echo "测速进程：$!"
```

一般需要几分钟；NPZ 解压、模型初始化和 CUDA 首次预热较慢时会更久。无需重装依赖或重新编译已正常工作的扩展。
本地断网或关闭 SSH 不影响 nohup 任务；AutoDL 实例本身需保持运行。

## 查看结果

```bash
tail -n 30 -F logs/speed_probe_v1.log
```

按 Ctrl+C 只退出日志查看。完成后：

```bash
cat work_dir/speed_probe_v1/summary.md
cat work_dir/speed_probe_v1/recommended_cli.txt
```

将 `summary.md` 内容发回即可。`results.json` 保存完整数值，`results.csv` 可用表格软件打开。
`recommended_cli.txt` 只是可追加到训练命令的加速参数，不是完整训练命令，也不是配置文件。

## 如何解释

- `train_samples_per_second` 包含取数据、搬运、真实前向/反向和优化器更新。
- `train_peak_mib` 是 PyTorch 峰值已分配显存，通常小于 nvidia-smi 的总占用。
- 整轮预计时间 = 按该 batch 实际保留的训练样本数 / 训练吞吐量 + 全部测试样本数 / 测试吞吐量。
- 时间预估不含 epoch 启动、checkpoint 写入和模型初始化；TensorBoard 写入用空写入器替代。它不是实测完整 epoch，采用前应再确认一整轮。
- `time_saved=20%` 表示预计耗时降为原来的 80%，相当于吞吐量提高 25%，不是吞吐量只提高 20%。
- 调大训练 batch 会改变每轮更新次数与 BatchNorm，TF32 会改变部分 FP32 运算。正式原版/改进版必须从头使用同一套新设置，不与旧设置的结果直接比较。
- 不改变帧数、数据量或模型层数，也不省略每轮评估来获得表面加速。GradScaler 和有限损失检查保留。

## 中断后继续

确认原测速进程已退出，再使用完全相同的参数加 `--resume`；已完成或已确定 OOM 的组会跳过。

```bash
nohup "$PWD/.venv/bin/python" -u tools/benchmark_autodl.py \
  --data-path /root/autodl-tmp/ntu60/ntu/NTU60_CS.npz \
  --output-dir work_dir/speed_probe_v1 --resume \
  >> logs/speed_probe_v1.log 2>&1 < /dev/null &
```

某一组不是 OOM 而是运行错误时，会保留此前结果并停止，错误细节在该组的 `worker.log`。

## 可选设置

加 `--include-batch32` 会增加 batch 32 测试，3080 Ti 可能 OOM；默认不测它。修改测速参数时请使用新的输出目录。
加 `--no-tf32-case` 可省掉 TF32 候选。`--steps 120` 可增加计时长度，降低短测噪声。
默认只测当前可见的 `cuda:0`，不要与训练并行。显式 `--allow-busy` 可绕过空闲检查，但并行负载会使结果不可靠。
