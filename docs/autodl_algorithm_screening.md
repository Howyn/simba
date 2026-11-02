# AutoDL 状态空间算法实验

实现日期：2026-11-02。数据：NTU RGB+D 60，X-Sub，joint 单流。

## 本次新增的两项独立候选

| 名称 | 代码中的改动 | 首轮范围 |
| --- | --- | --- |
| `joint_state`（A，优先） | 保留全局 Mamba，新增共享权重的逐关节时间状态支路；支路没有 D 跳连，只读出状态，以固定 0.1 倍融合 | 只改 l4 |
| `dual_scale`（B） | 共用输入投影，在正值 Δ 和 0.5Δ 下分别更新状态，以初值 0.1 的通道门融合；D、z 和输出投影各应用一次 | 只改 l4 |

两者都修改时间状态计算。CTR、FR、旧双向扫描、运动条件 Δ、B/C 归一化均关闭；不把 A 和 B 同时叠加。模型不会跨样本、人体槽位或片段携带状态。借鉴来源和研究边界见 [实验方案](ssm_algorithm_screening_plan_2026-11-02.md)。

## 1. 准备

4090、3080 Ti 都可以，整轮实验固定同一种 GPU 与同一环境。保留现有可运行的 `.venv`，不需要重新安装 PyTorch。初筛统一 FP32、TF32 关闭，batch 16、64 帧、seed 4321、5 轮预热、学习率 0.025、SGD。暂不附加之前的 BF16 加速参数。

```bash
cd /root/autodl-tmp/Simba-git
git pull --ff-only
source .venv/bin/activate
export OMP_NUM_THREADS=1
```

三个 `config/nturgbd-cross-subject/joint_algorithm_*.yaml` 文件的默认数据路径都是：

```text
/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz
```

若实际路径不同，开始任何检查前，将三个配置中 train/test 的 `data_path` 一起改成实际路径。配置使用 JSON 写法，属于合法 YAML；保留双引号和 JSON 布尔值。检查后不要改代码、数据或配方，否则需重做检查。

## 2. 先做 40 步真实数据检查

```bash
python tools/check_algorithm_signal.py joint_state
python tools/check_algorithm_signal.py dual_scale
```

每项只做 40 次交叉熵训练更新，不保存训练权重；在第 0、10、20、40 步测量修正比例和固定输入的 logit 差，在第 10、20、40 步记录任务梯度与权重衰减梯度。短检查以 `num_workers=0` 加载真实训练样本，减少诊断进程开销；正式训练仍使用固定的 8 workers 配方。

只有显示 `PASS` 才启动该候选的 10 轮训练。`先修模块` 或 `ERROR` 时把终端输出和对应 JSON 发回，先定位原因，不手动把检查结果改成通过。

检查内容：公共主干初始化与随机数流相同；所有梯度/输出有限；新增参数存在非零任务梯度；第 20、40 步有效修正比例在 `1e-4～1`；固定输入 logit 变化超过 `max(1e-6, 10×算子路径误差)`。候选 B 还核对正值 Δ 比例和状态保留因子。阈值是工程筛查规则，PASS 不代表准确率会提高，未通过也不能直接证明研究思路无效。

检查凭据保存在：

```text
work_dir/ntu60/xsub/algorithm_checks/joint_state_seed4321.json
work_dir/ntu60/xsub/algorithm_checks/dual_scale_seed4321.json
```

首次读取 NPZ 会计算文件指纹，可能需要等待；之后按文件大小、修改时间等核对后复用缓存。凭据同时绑定代码、配方、数据、GPU 和运行环境。

## 3. 通过后启动 10 轮初筛

两项均 PASS：

```bash
bash run_algorithm_screening.sh start10 --modes baseline joint_state dual_scale
```

只想先投入优先候选 A，或 B 尚未通过：

```bash
bash run_algorithm_screening.sh start10 --modes baseline joint_state
```

A 完成后，B 通过检查再单独追加：

```bash
bash run_algorithm_screening.sh start10 --modes dual_scale
```

脚本在后台依次运行，关闭 SSH 连接仍会继续；AutoDL 关机或释放实例会停止。不要同时运行旧训练命令。某组失败时队列停止，不会继续消耗后续组的预算。相同命令再次执行会跳过已完成且核验一致的组，未完成组从自己的 `runs-latest.pt` 恢复；没有可恢复检查点时拒绝覆盖已有目录。

```bash
bash run_algorithm_screening.sh status
bash run_algorithm_screening.sh logs --follow
bash run_algorithm_screening.sh results
bash run_algorithm_screening.sh results --json
```

`logs --follow` 只查看日志，按 Ctrl+C 退出查看不会停止后台训练。组目录分别是：

```text
work_dir/ntu60/xsub/algorithm_baseline_seed4321/
work_dir/ntu60/xsub/algorithm_joint_state_seed4321/
work_dir/ntu60/xsub/algorithm_dual_scale_seed4321/
```

每组含 `log.txt`、`console.log`、`config.yaml`、`run_manifest.json`、`runs-best.pt`、`runs-latest.pt`。队列日志在 `algorithm_screening_seed4321_queue/runner.log`。

本流程的基线同一配方只建一次，A/B 共同复用；不会每换一个模块再跑一次基线。历史实验若没有足以证明配方、数据和设备一致的记录，脚本不能仅凭一个 Top1 数值自动认证为本轮基线。若已有完整 20 轮、带本协议 `run_manifest.json` 且核验一致的基线，可在初筛前冻结复用：

```bash
bash run_algorithm_screening.sh reuse-baseline --directory /实际路径/已有基线目录
bash run_algorithm_screening.sh start10 --modes joint_state dual_scale
```

旧目录不会被改写；缺少凭据则拒绝，不会悄悄覆盖或重跑。切换 GPU 型号、种子、配方或代码后不能直接混用本轮结果；同一目录发生指纹不匹配时会报错。

## 4. 查看结果，再决定是否追加

`results` 同时给出 E10、最后 3 轮均值、前 10 轮最佳值、相对基线百分点差、领先轮数、训练时间和显存。不会把训练结束时重测的 best 模型错误计为第 1 轮。

`值得追加` 需满足其一：最后 3 轮均值领先 ≥0.5 个百分点、10 轮至少 6 轮领先且尾 3 轮至少 2 轮领先；或者领先 ≥0.3 且 <0.5 个百分点、至少 7 轮领先且尾 3 轮全部领先。其余正增益记 `观察`，均值不涨记 `暂缓`，信号检查失败记 `先修模块`。不按单轮峰值判断成功。

脚本输出 `Recommended single extension`，只允许其中一项追加。假设推荐 A：

```bash
bash run_algorithm_screening.sh extend20 --mode joint_state
```

若推荐 B，最后一个参数改为 `dual_scale`。若显示 `none`，不追加。该命令把基线和选中的候选都从各自第 10 轮 **latest** 继续到 20 轮；基线已有核验通过的 20 轮结果则直接复用。恢复优化器及 Python/NumPy/Torch/CUDA 随机数状态，拒绝从最佳轮次或缺少 RNG 状态的旧检查点冒充严格续训。

学习率在显示的第 15、19 轮下降。20 轮汇总报告 E20、18–20 均值、16–20 均值及 19–20 均值与基线差；第二次降学习率后只有两轮，不写成“降后 3–5 轮”。CUDA 非确定性算子仍可能引入差异，恢复随机数状态不等于逐位完全一致。

确认 20 轮趋势后，再对胜出项做独立配对种子。示例（先仅筛选同一个候选）：

```bash
python tools/check_algorithm_signal.py joint_state --seed 2026
bash run_algorithm_screening.sh --seed 2026 start10 --modes baseline joint_state
bash run_algorithm_screening.sh --seed 2026 results
```

新种子使用独立目录和自己的基线。这一步只查看 10 轮方向，不等于 20 轮多种子确认。正式多种子确认应先固定同一候选、全部种子与统一训练预算，连同不涨的种子一起报告；不要只把新种子中再次满足初筛门槛的结果纳入 20 轮均值。`extend20` 是初筛预算分配命令，仍执行该种子的初筛规则，不充当无条件的正式多种子确认入口。

没有通过多种子与完整训练验证前，只称初筛趋势。当前每轮评测使用官方测试集，因此这些结果属于探索性选择，最终评价应冻结方案后用独立验证选择及固定测试流程确认，不能将初筛挑选后的峰值宣称为无偏最终性能。

## 验证范围

本地 CPU 测试覆盖状态递推、形状、梯度、独立状态、初始化随机数流、门控参数衰减、检查凭据、结果判定、队列恢复和安全检查点。真实 CUDA 核函数、完整 Simba 与 NTU 数据需以上述 40 步检查为准；尚无本次候选准确率实测。
