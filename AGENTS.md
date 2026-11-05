# Simba research preferences

Read `EXPERIMENT_PROTOCOL.md` before preparing new training runs or comparisons.
The current default is the user's approved 10-epoch short protocol and measured
3080 Ti speed settings. Do not silently return to 15/25-epoch or fixed-0.005 runs.
Preserve the original Mamba switch-off path. The user now authorizes combination1:
Hyper-GCN spatial replacements at l4/l7/l10 plus DeGCN temporal replacements at
l5/l8. Run ONLY the combined 10-epoch variant first, reuse the prior verified
baseline, and leave single-module ablations for later. MTI/BiMamba stay disabled.
When requested, shut down AutoDL 300 seconds after this job exits; support cancel.
Training runs on the user's AutoDL instance; provide background/resume commands.
