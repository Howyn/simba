# 当前默认实验约定（用户于 2026-11-05 确认）

这是后续实验的默认设置；历史 MTI 配置只用于重现旧实验。

- NTU60 / X-Sub / joint，64 帧；空间 U-ShiftGCN 不改。
- 3080 Ti 单卡；训练 batch=24、测试 batch=64，BF16 autocast + GradScaler，TF32 开启。
- num_workers=8、pin_memory=true、prefetch_factor=2、persistent_workers=false。
- 默认 seed=4321；SGD、momentum=0.9、Nesterov=true、weight_decay=0.0004。
- **先跑 10 epoch 初筛**。不自动增加到 15/25 epoch。
- 预先确定学习率：第 1 轮 0.0125；第 2–6 轮 0.025；第 7–10 轮 0.0025。
- 原版和改进版从头使用完全相同的日程、种子、精度、batch 和数据增强。
- 观察第 8–10 轮。距该窗口均值最大偏离不超过 0.2 个百分点时才输出稳定均值；
  不稳定则给逐轮结果与范围，不强行输出稳定成绩。10 轮是初筛，不保证最终收敛。
- 同协议、同种子、同代码的基线可复用；换协议或代码不得混用旧基线。
- 一次只加一个有 2024–2026 年论文依据的模块。MTI 暂停。
- 原版通过 `use_mti=false, use_bimamba=false` 保留；方案 1 为 Vim v2 双向 SSM。
- AutoDL 使用 nohup 后台顺序跑基线再跑改进版；每轮保存完整恢复 checkpoint。
- 3080 Ti 短测速：上述加速设置相对 16/16、TF32 关闭预计节约 19.6% 时间。
  基线预估 11.12 分钟/轮；新模块成本另看实测日志，不保证相同耗时。
