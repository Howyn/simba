# 状态时序建模改进：多尺度上下文写入 P + 保留/写入独立控制 Q

实现日期：2026-11-02。数据：NTU RGB+D 60，**X-Sub**，**Joint** 单流，64 帧，骨架数据集 `NTU60_CS.npz`。

本轮只改**状态更新本身**，只接 **l4**，保留一套状态递推。P、Q 是设计代号；两者可单独开，也可同时开，组合就是“P＋Q”。

## 1 改了什么

原 selective-scan 的递推与读出：

    h_t = exp(Δ_t A) h_(t-1) + Δ_t B_t u_t
    m_t = C_t h_t
    y_t = (m_t + D ⊙ u_t) ⊙ SiLU(z_t)

本轮：

    h_t = exp(r_t Δ_t A) h_(t-1) + w_t Δ_t B_t^context u_t

| 代号 | 名称 | 代码开关 | 改的位置 |
| --- | --- | --- | --- |
| P | 多尺度上下文写入 | `ssm_context_layers: [4]` | `B_t → B_t^context` |
| Q | 保留/写入独立控制 | `ssm_retain_layers: [4]` | `Δ_t → r_t Δ_t`，写入强度 `w_t` |

### P：多尺度上下文写入（`context_write`）

对已编码的 `u`（`in_proj` + 因果卷积 + `x_proj` 之后，`d_inner=1250`）：

1. LayerNorm（无仿射）→ 低秩压缩 `d_inner → rank=16`；
2. 三路**深度可分离**时序卷积 `k=3,5,7`，双侧零填充，**不做首尾循环**；
3. 拼接 48 维 → `Linear → SiLU` 融合回 16 维；
4. 升维 `rank → d_state=16`，末层**小非零**初始化（`std = 1/sqrt(d_inner)`）；
5. 有界残差叠加到写入矩阵：

       B_t^context = B_t + 0.5 · tanh(residual_t)

（初版为 0.1；首轮 40 步实测后上调为 0.5，原因见 §5.1。）

C、Δ、A、D、z、`out_proj` 全部保留原值。首版只改“写入什么”，不同时改三种选择性参数。

参考：TSkel-Mamba 的多尺度时间交互（MTI）思路 <https://arxiv.org/html/2512.11503v1>。**适配边界**：原论文把上下文用在时间建模分支，这里把上下文用于**状态写入参数 B**，是新的适配规格，不能称为 MTI 复现；也没有复现其双向/空间 Transformer 组件。

### Q：保留/写入独立控制（`retain_write`）

    r = 0.25 + 0.75 · sigmoid(θ_r)   ∈ (0.25, 1)     θ_r 初始使 r = 0.5
    w = 2 · sigmoid(θ_w)             ∈ (0, 2)        θ_w 初始使 w = 1.0

`r`、`w` 都是**逐内部通道**的参数（各 1250 个），不是逐样本或逐帧；初始值就是方案要求的 `r=0.5, w=1`，所以**第一次前向**就延长了记忆时间，不必等一个弱融合门慢慢打开。

实现上复用现有扫描内核：把 `Δ` 乘 `r` 得到更长的 retention；把 `u` 乘 `w/r`，扫描内核的写入项于是恰好是 `w·Δ·B·u`，与 `r` 无关。`D` 跳连与 `z` 门控在扫描外、用**原始 u** 各做一次，所以跳连没有被改动。

- 两处 logit 都放进**零 weight decay** 组（`tools/ssm_optimizer.py`）。若让衰减作用于 logit，即使任务梯度为零，`r` 也会从 0.5 漂到 0.625，机制会被优化器悄悄改掉。
- 有界：`r` 不会低于 0.25（写入最多放大 4 倍），`w` 不超过 2。仍未越界的极端情况下 `u·(w/r)` 最大 8 倍，所以必须看状态能量。

参考：Naju 的独立记忆门思想，这里用我们自己的有界逐通道适配，不是其实现。

## 2 与基线的关系（可退回性）

- 基线路径完全不变：`MambaBlock` 在四个新开关都为 False 时仍构造原来的 `Mamba(d_model, d_state=16, d_conv=4, expand=2)`，参数名、形状、随机数消耗、前向都相同。
- 新参数在 `torch.random.fork_rng` 内初始化，公共骨干在同一 seed 下的初始化与数据顺序不变（40 步检查会核对这一点）。
- 关掉新机制（`context_enabled=False`、`retain_write_enabled=False`）即走原 fused 前向；`correction_enabled=False` 走“显式但没有修正”的路径，用来标定 fused 与显式算子的**浮点误差底**。
- `r=1, w=1, 无上下文` 时与基线递推逐位一致（CPU 测试 `tests/test_context_write.py` 覆盖）。
- 新增参数：P 约 2 万，Q 2×1250；相对主干可忽略，但**参数少不等于训练免费**：P 多两次小卷积 + 一次低秩升维，Q 只改逐通道标量，扫描次数不变。

## 3 预算与判据（预登记）

| 实验 | 轮数 | 目录 |
| --- | ---: | --- |
| 已有基线 | 复用首 5 轮，不新增训练 | 见 `reuse-legacy-baseline` |
| P 单独 `context_write` | 5 | `work_dir/ntu60/xsub/algorithm_context_write_seed4321/` |
| Q 单独 `retain_write` | 5 | `algorithm_retain_write_seed4321/` |
| P＋Q `context_retain_write` | 5 | `algorithm_context_retain_write_seed4321/` |
| 唯一赢家补到 10 轮 | +5 | 同目录续训 |

第一阶段最多新增 20 个训练轮次（三组各 5 轮 ＋ 赢家 5 轮）。5 轮内没有学习率衰减（warmup 5 轮，`step=[14,18]`），因此前 5 轮与旧 10/20 轮实验的**前 5 轮同配方**，历史基线的前 5 轮可以合法复用。

判据（`results5` 会自动判定）：

1. 第 3–5 轮平均 Top1 相对配对基线 **≥ +0.5 个百分点**；
2. 五轮中**至少四轮**领先，且第 3、4、5 轮**均**领先；
3. 组合若入选，还需比**最强单项**再高约 **0.2 个百分点**，否则优先保留单项。

不按单轮峰值判定，不因第 1–2 轮涨跌中途改学习率。5 轮仍在 warmup 内，这只是预算内的趋势证据，不是收敛结论。

## 4 运行步骤（AutoDL，4090/4080 Ti 均可）

```bash
cd /root/autodl-tmp/Simba-git        # 或你的仓库目录
git pull --ff-only
source .venv/bin/activate
export OMP_NUM_THREADS=1
```

数据路径写死在三个新配置里：`/root/autodl-tmp/ntu60/ntu/NTU60_CS.npz`。若不同，**三个配置一起改** train/test 的 `data_path`。

### 4.1 先跑 CPU 单元测试（约 20 秒，不需要 GPU）

```bash
python -m unittest discover -s tests -p "test_context_*.py" -v
python -m unittest discover -s tests -p "test_algorithm_*.py"
```

覆盖：递推公式与独立参考实现一致、`r=1/w=1` 退回基线、retention 是基线的 `r` 次幂、`w/r` 写入等价、边界不平移（无循环填充）、新参数梯度非零、增益有界、初始化不扰动骨干 RNG、增益不受 weight decay、5 轮判定规则与队列预检。

### 4.2 复用已有基线（关键，省 5 轮）

新模块会让所有模式的源码指纹变化，严格 manifest 校验会拒绝旧基线。`reuse-legacy-baseline` 用**可证明的替代核验**放行，并把你确认为真的项目写进 `baseline_reference.json`：

- 旧目录 `config.yaml` 的实际配方、seed、数据路径一致；
- 数据 NPZ 指纹一致；
- GPU 型号 / torch / CUDA 身份一致；
- 所有算法开关为**空**（确认它就是原 Simba）；
- 旧 checkpoint 的参数**名字与形状**与当前代码构造的原版网络**完全一致**（`tools/verify_baseline_architecture.py`）。

未被比较的是旧运行的源码指纹（因为新增模块必然改变它），这一点会打印在记录里，不会假装比过。

```bash
# 用你实际存在的目录，二选一
python tools/manage_algorithm_screening.py reuse-legacy-baseline \
  --directory work_dir/ntu60/xsub/full_baseline_20ep
# 或
python tools/manage_algorithm_screening.py reuse-legacy-baseline \
  --directory work_dir/ntu60/xsub/algorithm_baseline_seed4321
```

要求该目录至少有**前 5 轮**的 `log.txt` 评估记录。复用只读，不改旧文件。

如果不想复用，`start5 --modes baseline ...` 会先跑 5 轮基线（4090 上约 1 小时）。

### 4.3 40 步真实训练信号检查（每组一次，约 5–10 分钟）

```bash
python tools/check_algorithm_signal.py context_write
python tools/check_algorithm_signal.py retain_write
python tools/check_algorithm_signal.py context_retain_write
```

只有三个都显示 `PASS` 才启动对应组；`先修模块` 时先看下面的“不通过怎么办”。凭据写到 `work_dir/ntu60/xsub/algorithm_checks/<mode>_seed4321.json`，训练队列会强制校验。

检查内容：骨干初始化与 RNG 一致；loss/梯度/参数有限；新参数有非零任务梯度；第 20、40 步有效修正比例在 `1e-4～3`；固定输入 logit 变化超过 `max(1e-6, 10×数值误差底)`；机制不变量（retention 是否等于基线保留因子的 `r` 次幂、写入尺度是否等于 `w/r`、上下文残差是否非零且不超过 B 的一半）；**状态能量比**在 `[0.1, 10]`。

### 4.4 启动 5 轮队列

```bash
# 三个都通过（基线已复用）
bash run_context_stage.sh start5 --modes context_write retain_write context_retain_write
# 只想先投两个，或某一组还没通过检查
bash run_context_stage.sh start5 --modes context_write context_retain_write
```

后台顺序执行，SSH 断开不中断（关机/释放实例会中断）。同一 GPU 上不要与旧队列同时开。

```bash
bash run_context_stage.sh status
bash run_context_stage.sh logs --follow
bash run_context_stage.sh results5
```

某组失败时队列停止、不消耗后续预算；相同命令重跑会跳过已完成组，未完成组从自己的 `runs-latest.pt` 续跑。

### 4.5 看结果，再决定补到 10 轮

`results5` 输出 E5、第 3–5 轮均值、前 5 轮最佳、相对基线的百分点差、领先轮数/5、末三轮逐轮差、训练小时与峰值显存，并给出 `Recommended single extension`。

```bash
bash run_context_stage.sh extend10 --mode <推荐的那个>
```

它把基线（若只有 5 轮）与赢家都从各自**第 5 轮 latest** 严格续到 10 轮（保留优化器与 RNG 状态，拒绝用 best 冒充）。复用外部基线若已有 10/20 轮则直接比较、不改旧目录。10 轮判据沿用原协议（末三轮均值 ≥ +0.5 且 10 轮中 ≥6 轮领先等）。

### 4.6 只推理的状态关断诊断（用已训权重）

```bash
python tools/ablate_context_checkpoint.py \
  --checkpoint work_dir/ntu60/xsub/algorithm_context_retain_write_seed4321/runs-best.pt
```

五种临时干预：完整、关上下文、关保留/写入增益（`r=w=1`）、两者都关（保留显式算子路径，用于看浮点误差底）、状态读出置零（保留 D 与 z）。输出 Top1、预测翻转数、最大 logit 差；全部在内存中恢复并用 `state_dict` 逐位核对，不写文件。

判读：若“关增益”或“状态读出置零”几乎不掉点，说明分类主要靠 D 跳连/点式变换，新机制没有真正参与决策；此时不要追加 10 轮。

## 5 必须观察的量（不能只看准确率）

| 量 | 期望 | 说明 |
| --- | --- | --- |
| `retention` / 基线保留因子 | 等于 `r` 次幂 | 否则 Δ 被乘错位置 |
| `u_write_scale` | 等于 `w/r` | 保证写入强度确实独立于 r |
| 状态能量比 | 约 1–3（r=0.5 时约 2） | 输出变大不等于效果变好；超过 10 视为失控 |
| 上下文残差占比 | 2%–20% | 太小查初始化/尺度，太大压过原写入 |
| `Δ` 有效值 | 与基线同量级 | 乘 `r` 后不应出现数值下溢 |

### 5.1 首轮 40 步实测（seed 4321）与尺度修订

三组各做 40 次真实训练更新后的记录：

| 组 | 初始 corr | 第 40 步 corr | 状态能量 | 机制不变量 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| `retain_write` | 1.21e-01 | 3.40e-03 | 1.52 | 全过（`ret=0.500`、`write=1.000`、幂误差 1.3e-07） | PASS |
| `context_retain_write` | 1.21e-01 | 6.54e-03 | 1.54 | 全过 | PASS |
| `context_write`（尺度 0.1） | 2.53e-04 | 1.04e-04 | 1.01 | 全过（上下文残差占比 1.3%） | FAIL：corr 低于 1e-4，logit 低于噪声底 |

`context_write` 失效的原因是架构性的，不是实现错误：l4 的 Mamba 输出里 `D·u` 跳连与 `z` 门控占主导，状态读出只占一小部分，因此“只改写入矩阵”的机制初始可观测信号必然小。三个相互独立的证据：

1. 同样是**改状态更新**的旧候选 `dual_scale`，corr 为 7.25e-05，同样低于门槛；
2. **加法支路** `joint_state` 不经过状态递推，corr 为 1.3e-03，通过；
3. Q 的初始信号大 12%，因为它从第一次前向就改变了整个历史累积，而不只是写入矩阵。

因此按“查实现/尺度”的预登记处置，把 P 的 `ssm_context_scale` 由 0.1 上调到 0.5（单项与组合**同一尺度**，否则 2×2 对比不成立），并按实测比例预期：上下文占比 1.3%→约 6.5%，corr 约 2e-04，logit 同比放大 5 倍。上调后必须重跑 `context_write` 与 `context_retain_write` 两组检查；`retain_write` 的配置未变，凭据仍有效。

**不要**在没有新证据的情况下继续往上加尺度：0.5 是本次唯一一次调整，若仍不过，结论是“在当前 l4 接入点上，只改写入矩阵的机制信号不足”，改为只在组合中保留 P，或改接 l7/l10 另开一轮。

## 6 不通过怎么办（按顺序，不要一次改多项）

1. **增益越界/状态能量 >10**：把 `ssm_retain_init` 从 0.5 调到 0.75（记忆只延长约 1.33 倍），单改这一项重跑检查。
2. **上下文残差 <1% 且修正比例 <1e-4**：把 `ssm_context_scale` 从 0.5 调到 1.0（0.1→0.5 已按 §5.1 执行过一次），或 `ssm_context_rank` 16→32。
3. **梯度全为零**：先查 `block.mamba` 是否真的是 `ContextWriteMamba`、`l4` 是否被 `ssm_context_layers`/`ssm_retain_layers` 命中，不要靠放大 gain 掩盖。
4. **只有组合涨、单项不涨**：按预登记规则优先保留单项，除非组合超出 0.2 个百分点。

改动配置后 `config_sha256` 变化，队列会要求干净的目录，不会混用新旧结果。

## 7 结论边界（诚实说明）

- P、Q 都是**适配**，不是论文实现复现；TSkel-Mamba 在 NTU120 X-Sub joint 上 MTI +0.9 个百分点（T-Scan 85.8→86.7）不能换算成这里的收益。
- `r`、`w` 是**静态逐通道**参数：它们只给每个通道一个可学的“记忆时间/写入强度”偏置，不随输入变化。若单项无效，下一步才是让 `r_t`、`w_t` 由 `u` 的低秩投影生成（首轮先不加这个自由度）。
- 5 轮、单 seed、每轮直接评估官方测试集：属于**探索性筛选**，测试集已参与选择，不能称为无偏最终性能。要下结论需要冻结方案 + 配对多 seed（同 seed 的基线必须重新配对，不能拿别的 seed 的基线顶替）。
- 显存与耗时以日志实测为准；P 增加约两次小卷积，Q 不增加扫描次数。

## 8 来源

1. Gu and Dao. Mamba. <https://arxiv.org/abs/2312.00752>
2. TSkel-Mamba，多尺度时间交互（MTI），Table III 的 NTU120 X-Sub joint 消融。 <https://arxiv.org/html/2512.11503v1>
3. Naju 的独立记忆门思想（有界适配为本仓库自定义）。 <https://arxiv.org/html/2607.21000v1>
4. Mamba-3 / Gated DeltaNet-2：完整迁移涉及新的递推与内核，列为后备方向，本轮不做。 <https://arxiv.org/html/2603.15569v1>
