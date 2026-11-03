# Simba 负特征值扩展 10 epoch 筛选

本轮只在现有 Simba 基线的 Mamba 状态递推中引入负特征值扩展（ICLR 2025 Oral），目标是把"机制是否生效"和"是否涨点"分开验证。两组各跑 10 epoch，先过算子自检，再看精度；10 epoch 只代表早期训练表现，不宣称最终性能。

| 组别 | 配置 | scan_mode | 默认结果目录 |
| --- | --- | --- | --- |
| A 基线 | `config/nturgbd-cross-subject/joint_autodl.yaml` | baseline | work_dir/ntu60/xsub/simba_joint |
| B 改进 | `config/nturgbd-cross-subject/joint_autodl_negative.yaml` | negative | work_dir/ntu60/xsub/simba_joint_neg_eig |

## 一 方法依据

论文：Grazzi R, Siems J, Zela A, et al. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. ICLR 2025.
<https://proceedings.iclr.cc/paper_files/paper/2025/file/5a0ce3abb720b740419e193c87afd080-Paper-Conference.pdf>

Mamba 的离散状态递推为

```
h[t] = a[t] * h[t-1] + delta[t] * B[t] * u[t],   a[t] = exp(delta[t] * A)
```

因为 `A = -exp(A_log) < 0` 且 `delta > 0`，基线里 `a[t] ∈ (0, 1)` 恒为正。负特征值扩展只替换状态保留项：

```
a[t] = 2 * exp(delta[t] * A) - 1  ∈ (-1, 1)
```

`a[t] < 0` 的条件是 `delta * |A| > ln 2`。注意这**不是**改 A 的符号（A 本来已经是负的），也不是输出乘负数；改的是连续系统离散化后的衰减映射本身。

不新增任何可学习参数，不改变张量形状、优化器、数据管道，计算量不变（每步仍是 1 次 `exp2f`）。本方法引用原论文，不作为本文原创算法。

### 初始化时的负值比例（重要）

`A = -[1, 2, ..., 16]`，`delta` 在每个通道上按对数均匀取 `[0.001, 0.1]`（中位数约 0.01）。`a < 0` 要求 `delta * n > ln 2 ≈ 0.693`，即 `delta > 0.693 / n`：

| 状态维 n | 8 | 10 | 12 | 14 | 16 |
| --- | --- | --- | --- | --- | --- |
| 该维初始化即为负值的概率 | 3% | 8% | 12% | 15% | 18% |

`n ≤ 6` 时在初始化区间内不可能为负。各维平均下来，**初始化时约 6%** 的 `(d, n)` 组合是负转移，不是"一半"。

两个直接推论：

1. 改进组与基线从初始化起动态就不同（同一个 `delta` 下 `2q-1 < q` 恒成立，所有状态的衰减都更快），但幅度有限：随机权重下 Mamba 输出与基线只差约 1e-3。所以前几轮曲线完全可能先落后再追，判断要看趋势与 E8–E10。
2. `scan_statistics.py` 在**未训练**模型上看到的 `frac<0` 预期是几个百分点，与上表一致；接近 0 说明实现或初始化有问题。训练后 `delta` 通常会变大、比例随之上升，可用 `--checkpoint` 复查训练后的模型。

## 二 与最初设想的差异：不新写一套 CUDA 扩展

原方案要求把负特征值前向/反向编译成独立扩展（`csrc/negative_selective_scan/`）。实际阅读算子后确认**没有必要**，原因是 Mamba 的 `selective_scan` 内核并不是"指数衰减专用"：

* 块内扫描是 cub `BlockScan` 对 `(a, b)` 做结合运算 `(a1,b1)⊗(a2,b2) = (a1*a2, a2*b1 + b2)`，对任意 `a_t` 都成立；
* 跨 chunk 由 `smem_running_prefix` 把上一步的前缀状态带过去；
* 尾部 padding 用 `(1, 0)` 作为递推恒等元，与 `a_t` 取值无关。

所以前向只需把 `exp2f(...)` 换成 `2 * exp2f(...) - 1`，且对负值、跨 chunk、非整块长度都精确。

反向需要重写，且不能靠除法还原前一时刻状态（`a_t` 可以接近 0）。设 `q = exp(delta * A)`：

* `∂(2q-1)/∂delta = 2q * A`，`∂(2q-1)/∂A = 2q * delta`；
* 核心里已有的量 `a = h[t] - b[t] = a[t] * h[t-1]`，负特征值下系数要变成 `2q * h[t-1] = (h[t] - b[t]) + h[t-1]`；
* `h[t-1]` 在同一线程内就是 `thread_data[i-1].y`，线程边界处用一小块共享内存 `smem_h_prev`（每个线程段末的前缀状态）传递。

实现选择：在**同一个扩展**里加运行时 `bool negative` 分派，而不是新扩展。理由是 baseline 分支的算术没有被改动（同一个 `exp2f`、同样的 `a` 系数），历史基线仍可直接比较；新扩展会把编译时间翻倍，且两份 `.so` 容易不同步。代价是内层循环多一个 warp 一致的分支。

`selective_scan_cuda.negative_scan_supported` 由 C++ 侧暴露，Python 侧据此判断是否已经重新编译：用旧 `.so` 跑 `scan_mode=negative` 会直接报错，不会静默退回基线算子。

## 三 代码改动

| 文件 | 改动 |
| --- | --- |
| `csrc/selective_scan/selective_scan.h` | `SSMParamsBase` 增加 `bool negative` |
| `csrc/selective_scan/selective_scan_fwd_kernel.cuh` | 转移系数改为 `2 * exp2f(...) - 1`（实数分支） |
| `csrc/selective_scan/selective_scan_bwd_kernel.cuh` | 转移系数、反向扫描的转移值、梯度系数 `2q·h[t-1]`；新增 `smem_h_prev` 及其共享内存 |
| `csrc/selective_scan/selective_scan.cpp` | `fwd`/`bwd` 增加 `negative` 参数，复数 A 报错，暴露 `negative_scan_supported` |
| `mamba_ssm/ops/selective_scan_interface.py` | 透传 `negative`；新增旧 `.so` 的显式报错 |
| `mamba_ssm/ops/scan_reference.py` | 新增纯 PyTorch 参考递推（仅依赖 torch，可在 CPU 上测试） |
| `mamba_ssm/modules/mamba_simple.py` | 新增 `scan_mode`，快速路径/慢速路径/`step` 均分派 |
| `mamba_blocks.py`、`model/Hyperformer.py` | `scan_mode` 从 Model 传到 l1–l10 的 MambaBlock |
| `main.py` | 检查点记录 `scan_mode`；`--weights`/`--resume` 时模式不匹配直接报错 |
| `config/.../joint_autodl_negative.yaml` | 只改 `work_dir` 和 `scan_mode` 的配置 |
| `run_negative.sh` | 两组训练入口 |
| `tools/check_negative_scan_cuda.py` | GPU 自检：原生算子 vs 参考递推 |
| `tools/scan_statistics.py` | 真实数据上 l1–l10 的系数分布与梯度范数 |
| `tests/test_negative_scan.py` | CPU 单元测试 |

## 四 自检（训练前必须全过）

```bash
cd /root/autodl-tmp/Simba-main        # 换成实际仓库路径
source .venv/bin/activate             # 若有虚拟环境
git pull --ff-only origin main
git rev-parse --short HEAD

# 必须重新编译扩展，否则 scan_mode=negative 会直接报错
rm -rf build
python setup.py build_ext --inplace
```

```bash
python tools/check_backward_derivation.py           # 纯 numpy，验证反向公式，不需要 GPU/PyTorch
python -m unittest tests.test_negative_scan -v      # CPU，秒级
python tools/check_negative_scan_cuda.py            # GPU，必须全过
```

`check_backward_derivation.py` 用中心差分独立校验核心反向用到的系数（`2q·h[t-1] = (h[t]-b[t]) + h[t-1]`），两种模式都应输出 `ok`，相对误差约 1e-10。它验证的是公式，不验证 CUDA 实现。

`check_negative_scan_cuda.py` 依次检查：扩展是否带负特征值分支；长度为 1 / 17 / 64 / 130 / 300（padding、单 chunk、多 chunk）时两种模式的前向与**全部**梯度是否与参考递推一致；`d_inner=1250` 的真实尺寸；参考实现的 float64 有限差分；Mamba 模块级（同种子参数完全相同、negative 输出确实与 baseline 不同）；整个 Simba 模型 64/32/16 帧的前向反向。

任一项失败都不要开始训练，把完整输出发回来。预期最后一行是 `All negative-eigenvalue checks passed ...`。

机制统计（在真实数据上确认负值比例，不读训练结果、不消耗随机数）：

```bash
python tools/scan_statistics.py \
  --config config/nturgbd-cross-subject/joint_autodl_negative.yaml --batches 4
```

输出每层的 `min/max/mean/frac<0/frac|a|<1e-3` 与 `A_log`、`dt_proj` 的梯度范数。基线组同一命令会显示 `frac<0 = 0`，可作为对照。

## 五 统一条件

| 项目 | 设置 |
| --- | --- |
| 数据 | NTU RGB+D 60，X-Sub，Joint，`NTU60_CS.npz`，64 帧 |
| 种子 / 批大小 | 4321 / 16 |
| 优化器 | SGD，lr 0.025，weight decay 0.0004，nesterov |
| 学习率计划 | warm-up 5 epoch，衰减点 40（10 epoch 内不衰减，与既有基线口径一致） |
| 精度 | 先 FP32 |
| GPU | 3080 Ti 或 4090，同一对照必须同一张卡、同一环境 |
| 轮数 | 10（只有结果值得继续才补到 20） |

两个配置除 `work_dir` 和 `scan_mode` 外完全一致。

## 六 训练命令

```bash
bash run_negative.sh negative --num-epoch 10
bash run_negative.sh baseline --num-epoch 10
```

结果分别写到 `work_dir/ntu60/xsub/simba_joint_neg_eig/` 和 `work_dir/ntu60/xsub/simba_joint/`。若目录已存在，程序会询问是否删除，先确认里面没有要保留的历史结果。

建议同一天、同一张卡、依次（不并行）跑完两组，避免驱动/温度/占用差异混进对比。想同时开两组就用不同 GPU 并在命令后加 `--device 1`。

评估某一组的最佳检查点：

```bash
bash test.sh work_dir/ntu60/xsub/simba_joint_neg_eig/runs-best.pt \
  config/nturgbd-cross-subject/joint_autodl_negative.yaml
```

模式不匹配时 `main.py` 会直接报错，不会用错算子打分。

## 七 结果记录（待填）

| 指标 | 基线 A | 改进 B | 差值 |
| --- | --- | --- | --- |
| E5 测试集 Top-1 | 待实验 | 待实验 | 待计算 |
| E10 测试集 Top-1 | 待实验 | 待实验 | 待计算 |
| E8–E10 均值 | 待实验 | 待实验 | 待计算 |
| 平均每 epoch 时间 | 待实验 | 待实验 | 待计算 |
| 峰值显存 | 待实验 | 待实验 | 待计算 |
| 负值比例 / 梯度范数 | 0 | 待实验 | — |
| 训练是否出现 NaN/Inf | 待实验 | 待实验 | — |

历史参考：上一轮 10 epoch 基线（同协议、不同代码版本）E10 为 74.82。是否复用取决于参数、数据、环境是否完全一致；最稳妥是同轮重跑基线。

## 八 判定规则（预注册）

硬门槛（任一不满足即停止）：

1. `check_negative_scan_cuda.py` 全部通过；
2. `scan_statistics.py` 中 l1–l10 的负值比例大于 0，且 `A_log`、`dt_proj` 梯度非零；
3. 10 epoch 训练无 NaN/Inf，loss 正常下降。

涨点门槛（沿用仓库既有规则）：

* E10 Top-1 相对同轮基线 **≥ +0.5**，且 E8–E10 均值 **≥ +0.5**：认为值得补到 20 epoch，并在预算允许时加种子；
* 差距很小、曲线交叉：如实记录为"机制生效但收益未证实"，不延长、不加种子；
* 明显落后或不稳定：停止，分析原因（负值比例过高？某些层退化？），再决定是否只改部分层。

不允许只挑某一轮最有利的结果，也不允许在 10 epoch 内换学习率计划。

## 九 已知风险与后续

* **本文档的 CUDA 补丁在编写环境里无法编译验证**（本机无 nvcc、无 GPU）。所以第四节的自检是唯一闸门，第一次务必先跑它，失败就把完整输出发回来。
* 论文中部分 Mamba 语言任务在负特征值下退化，动作识别不保证涨点；机制生效与精度提高是两件事。
* 负值比例取决于 `delta` 的分布。若统计显示几乎所有系数都为负（例如 >90%），状态相当于每步翻转，可能破坏骨架序列的平滑性；此时下一步应做"只改 l4/l7/l10"的分层消融，而不是继续加训练预算。
* 本方案不改 `A_log` 初始化、不改 `d_state`、不加门控，因此若 10 epoch 无提升，结论只能是"仅换衰减映射不足以涨点"，不能推广为"负特征值方法无效"。
