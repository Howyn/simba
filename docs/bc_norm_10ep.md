# B/C Norm：只跑一个 10 轮候选

这次只训练 `ssm_bc_norm`：在 `l4` 的 SSM 内部，对输入相关的 B/C 状态投影作 RMS 归一化，再使用可学习的 gain 与 bias；原双向扫描、运动 Δ、CTR-GC、FR-Head 均关闭。gain 和 bias 均从 `0.1` 开始，eps 为 `1e-6`，四组参数不施加权重衰减。这是针对当前 Mamba 实现的保守适配，并非完整复现 Mamba-3；不能保证涨点。

候选从头训练，不载入旧 baseline 或组合模型权重。数据、增强、batch=16、seed=4321、SGD、FP32、TF32、学习率和 warmup 均沿用已有 `joint_baseline_20ep.yaml` 的前 10 轮。保留 `step: [14, 18]`，所以这 10 轮不会发生阶梯降学习率。若把 step 压缩到 10 轮内，就需要另跑相同日程的 baseline，不能继续使用旧的前 10 轮作公平参考。

## 更新和运行

在 AutoDL 项目目录、已有 `.venv` 中执行。先结束仍在占 GPU 的旧训练，再启动新候选。以下 CUDA 检查和 smoke 只校验实现；只有 `start` 会训练。

```bash
cd /root/autodl-tmp/Simba-git &&
source .venv/bin/activate &&
source /etc/network_turbo &&
export OMP_NUM_THREADS=1 &&
git pull --ff-only origin main &&
python tools/check_bc_norm_cuda.py &&
python tools/smoke_academic_model.py ssm_bc_norm &&
bash run_bc_experiment.sh start
```

`start` 返回后，训练在独立后台会话运行，断开 SSH 或本地断网不影响它。后台队列只包含这一组 10 轮实验。它会拒绝重复启动和覆盖没有恢复点的未完成目录。云实例关机、重启、释放或进程被杀仍会终止训练；重新执行 `start` 可从该候选的 `runs-latest.pt` 恢复，已完成则跳过。启动使用当前激活环境里的 Python，不会切换到系统 Python。

```bash
# 队列状态和当前指标
bash run_bc_experiment.sh status

# 持续查看训练输出；Ctrl+C 只退出查看，不会停止训练
bash run_bc_experiment.sh logs --follow

# 看最近 80 行，不持续跟踪
bash run_bc_experiment.sh logs --lines 80

# 任何时候都可查看；不会训练
bash run_bc_experiment.sh results
```

## 结果与对比

新结果目录为 `work_dir/ntu60/xsub/full_ssm_bc_norm_10ep/`，后台队列状态与启动日志位于 `work_dir/ntu60/xsub/ssm_bc_norm_10ep_queue/`。`runs-best.pt` 为该候选最佳权重，`runs-latest.pt` 用于恢复，`log.txt` 为逐轮指标，`console.log` 包含控制台错误和进度。

结果页同时显示 `baseline_first10` 和 `ssm_bc_norm`，以及两者最佳 Top1 的百分点差。参考行只读取既有 `full_baseline_20ep/log.txt` 的第 1–10 轮；你贴出的该区间最佳值是第 10 轮 `74.99%`。第 19 轮的 `86.1232%` 和日志末尾的 `Best accuracy` 不会混入这个 10 轮对比，也不读取 baseline 权重。旧日志缺失时参考显示空值，需要把原日志放回原目录后再看结果。

10 轮用于低成本筛选，单次训练的早期优势不能说明最终收敛优势。先看同样 10 轮的最佳和第 10 轮 Top1，再决定是否值得投入后续训练；不要把 10 轮结果写成完整训练成绩。

原 `run_full_experiments.sh` 默认仍为原有 20 轮四组入口，本次请使用 `run_bc_experiment.sh`。
