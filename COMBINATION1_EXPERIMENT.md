# 组合一：Hyper-GCN 空间模块 + DeGCN 时序模块

用户最新授权：先只训练组合版 10 epoch，与既有 short10 基线比较；不自动重跑基线，
不自动启动空间/时序单模块消融。单模块开关只为后续消融保留。

## 论文与来源

1. **Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action
   Recognition with Virtual Connections**，ICCV 2025，正式论文。
   [论文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Adaptive_Hyper-Graph_Convolution_Network_for_Skeleton-based_Human_Action_Recognition_with_ICCV_2025_paper.html)
   [作者实现](https://github.com/6UOOON9/Hyper-GCN/blob/cefe70b9c62ab825722cc8ece3fa26ecf5c334a1/model/hypergcn_base.py)
   来源版本：`cefe70b9c62ab825722cc8ece3fa26ecf5c334a1`。
   移植 HYPERGC 和 virtual_ensemble 图构造，未移植完整网络。
2. **DeGCN: Deformable Graph Convolutional Networks for Skeleton-Based Action
   Recognition**，IEEE Transactions on Image Processing，2024，正式论文。
   [论文](https://discovery.ucl.ac.uk/id/eprint/10189813/)
   [作者实现](https://github.com/WoominM/DeGCN_pytorch/blob/a0a261fbf4369095083a7b22fa371fb9448f69a8/model/modules.py)
   来源版本：`a0a261fbf4369095083a7b22fa371fb9448f69a8`。
   移植 DeTGC 和四分支 MultiScale_TemporalModeling，eta=4。空间 DeSGC 未完整公开，未采用。
   MIT 来源许可见 `licenses/DeGCN-MIT.txt`。

这两个来源的收益不能直接当作 Simba 上的预期收益；本实验不声称复现两篇论文的完整模型。

## 实际插入位置

- `use_hypergcn=true`：只替换 l4/l7/l10 的外层 `shift_gcn`，216→216。
  使用8头、3个虚拟关节、9邻居；从现有 NTU 邻接矩阵恢复无向物理连接，并按作者
  virtual_ensemble 规则加入虚拟关节。虚拟关节只在模块内部存在，输出仍为25关节。
- `use_detgc=true`：只替换 l5/l8 的 `tcn1`，保留 stride=2，即64→32→16帧。
  两个可变形时间分支（初始 dilation=1/2）与池化、1×1分支拼接，输出216通道。
  时间偏移是可学习共享参数，不是逐样本偏移预测网络。
- U形压缩/解码、跳连、625维原版 Mamba、池化、分类器和交叉熵不改。
- 两开关默认 false；false 时原 forward、参数和初始化随机流不变。
  新模块在原模型初始化完成后通过 fork_rng 构造，避免影响保留权重和后续数据随机流。
- MTI / BiMamba 与本组合互斥，不混入历史模块。

数值适配：超图距离、归一化与邻接矩阵聚合，以及时间插值的位置/权重在 FP32 下计算，
其余投影保留全局 BF16 autocast；避免用 BF16 做近邻距离和分数位置计算。
时间长度使用实际输入 T，边界使用作者实现的 clamp/线性插值；额外支持奇数长度。
距离使用直接欧氏 cdist，避免默认矩阵乘法距离路径的消减误差。

这是基于官方代码的 NTU 适配基线，不声称补齐了原论文未执行的部位门控。

## 锁定训练设置

全部继承 `config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml`：
NTU60 X-Sub joint / 64帧 / seed4321 / SGD momentum0.9 Nesterov / wd0.0004；
train24、eval64、BF16+GradScaler、TF32、workers8、pin_memory。
学习率第1轮0.0125、第2–6轮0.025、第7–10轮0.0025，总计10轮。
观察8–10轮，距窗口均值最大偏离<=0.2个百分点才报告稳定均值。
所有结果来自官方 test split，属于初筛，非独立开发验证集。

旧基线只读使用：
`work_dir/ntu60/xsub/bimamba_short10_v1/baseline_seed4321/results.json`。
需要完整1–10轮记录、相同 run_config、相同逐轮学习率，且父目录的
`source_revision.txt` 为 `282f095766ff9d60a86c9199fe19bf28e32122ab`。
缺失或不匹配就报错，不伪造来源文件、不从日志手工补成“已验证”结果，也不自动重训。
沿用原预处理数据；旧运行没有数据内容哈希，基线校验不能证明同路径文件未被替换。

## AutoDL 启动

```bash
cd /root/autodl-tmp/Simba-git
git pull --ff-only
mkdir -p logs
nohup "$PWD/.venv/bin/python" -u tools/run_combination1.py \
  --baseline-dir work_dir/ntu60/xsub/bimamba_short10_v1/baseline_seed4321 \
  --output-root work_dir/ntu60/xsub/combination1_short10_v1 \
  --shutdown-on-exit --shutdown-delay 300 \
  > logs/combination1_short10_v1.log 2>&1 < /dev/null &
echo "启动进程：$!"
```

默认 variant=combined，只训练组合版。无需编译新 CUDA 算子；继续使用已有 baseline scan。
启动自动进行参考输出/梯度检查、历史开关关闭路径校验，以及24批 BF16 完整反向检查。
它们只做短检查，不跑额外训练轮数；有错误则停止。

本地 CPU PyTorch 2.5.1 已通过14项数值/梯度/控制逻辑及既有 short10 回归检查。
完整模型 CUDA 检查本地未执行，启动时在 AutoDL 执行；未在本地或云端代跑训练。

**作业成功或报错退出后都等待300秒关机**，省去失败后的空转费用。
使用 [AutoDL 官方方式](https://www.autodl.com/docs/save_money/) `/usr/bin/shutdown`，
不使用可能不兼容的宿主机 shutdown 参数。不会在第一个 epoch 后关机。
若镜像提供没有 shebang 的文本 shell 包装器、直接执行报 ENOEXEC，脚本会先用
`bash -n` 检查语法，再通过 `/bin/bash /usr/bin/shutdown` 执行；不对二进制或权限错误盲目重试。
若另一 GPU 任务出现，或 GPU 空闲查询失败，脚本跳过关机并写入原因。
代码不会在本地 Windows 执行关机。

查看实时日志（Ctrl+C只退出查看）：

```bash
tail -n 30 -F /root/autodl-tmp/Simba-git/logs/combination1_short10_v1.log
```

查看逐轮结果：

```bash
grep -aE 'BASELINE REUSED|PREFLIGHT|START combined|EPOCH_RESULT|COMPARISON RESULT|COMPLETE|FAILED|SHUTDOWN' \
  /root/autodl-tmp/Simba-git/logs/combination1_short10_v1.log | tail -n 25
```

取消本次/未来本输出目录的自动关机（训练中或倒计时中均可）：

```bash
touch /root/autodl-tmp/Simba-git/work_dir/ntu60/xsub/combination1_short10_v1/KEEP_RUNNING
```

这是脚本内倒计时，`shutdown -c` 不能取消它。取消文件不会停止训练。
再次启用关机，需要在重启脚本前删除该标记。

续跑：重新开机后执行相同启动命令，使用同一代码提交/目录/配置。
自动加载 `combined_seed4321/runs-latest.pt`，恢复已完成 epoch 的模型、优化器、
GradScaler和随机数状态；当前未完成 epoch 重跑，不额外增加10轮。
如果第1轮完成前出错，没有 checkpoint，应使用新的 output-root 重新开始。
严格恢复会拒绝不同代码提交。无需断网重启，SSH断开不影响 nohup。

输出：

- `combined_seed4321/launch_config.yaml`：完整实际配置。
- `combined_seed4321/log.txt`、`metrics.csv`、`results.json`。
- `combined_seed4321/runs-latest.pt`、`runs-best.pt`、`runs-final.pt`。
- `comparison_combined_seed4321.md/.json`：与旧基线逐轮比较，不稳定时不强报均值。
- `shutdown_status.json`：关机倒计时、取消或失败原因。

后续只有用户确认后才运行 `--variant spatial` 或 `--variant temporal` 消融。
