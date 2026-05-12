#### tmux训练
``` bash
tmux new-session -d -s navida_qwen3_rxr_r2r_envdrop_scalevln "/bin/bash -lc 'cd /data1/code/embAI_sup/awzy/NAVIDA && bash scripts/train_qwen3_r2r_rxr_envdrop_scalevln_4gpu.sh > result/qwen3vl4b_r2r_rxr_envdrop_scalevln_4gpu.log 2>&1'"

```

### qwen3vl-4b rxr-r2r-envdrop-scalevln训练测试  val_seen

#### ckpt1000
``` bash
Success rate: 58/778 (0.075)
Oracle success rate: 107/778 (0.138)
SPL: 44.852/778 (0.058)
Distance to goal: 9.458
Path length: 8.479
```

#### ckpt7600
``` bash
Success rate: 164/778 (0.211)
Oracle success rate: 204/778 (0.262)
SPL: 154.234/778 (0.198)
Distance to goal: 8.616
Path length: 7.191
```

#### ckpt14400
``` bash
Success rate: 188/778 (0.242)
Oracle success rate: 228/778 (0.293)
SPL: 177.664/778 (0.228)
Distance to goal: 8.500
Path length: 7.435
```
