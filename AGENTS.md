# Simba research preferences

Read `EXPERIMENT_PROTOCOL.md` before preparing new training runs or comparisons.
The current default is the user's approved 10-epoch short protocol and measured
3080 Ti speed settings. Do not silently return to 15/25-epoch or fixed-0.005 runs.
Keep spatial GCN unchanged, preserve the original Mamba switch-off path, and
enable only one research module per experiment. MTI is paused, not combined.
Training runs on the user's AutoDL instance; provide background/resume commands.
