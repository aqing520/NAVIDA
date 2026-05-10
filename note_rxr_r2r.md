#### tmux训练
``` bash
tmux new-session -d -s navida_qwen3_rxr_r2r "/bin/bash -lc 'cd /data1/code/embAI_sup/awzy/NAVIDA && bash scripts/train_qwen3_r2r_rxr.sh > result/qwen3_r2r_rxr_tmux.log 2>&1'"

```

### qwen3vl-4b rxr-r2r训练测试  val_seen

#### ckpt200
``` bash
Success rate: 15/163 (0.092)
Oracle success rate: 27/163 (0.166)
SPL: 12.262/163 (0.075)
Distance to goal: 8.947
Path length: 7.669
```

#### ckpt400
``` bash
Success rate: 21/106 (0.198)
Oracle success rate: 26/106 (0.245)
SPL: 19.552/106 (0.184)
Distance to goal: 8.848
Path length: 8.316
```

#### ckpt600
``` bash
Success rate: 30/169 (0.178)
Oracle success rate: 45/169 (0.266)
SPL: 27.012/169 (0.160)
Distance to goal: 8.377
Path length: 8.159
```


#### ckpt1000
``` bash
Success rate: 169/778 (0.217)
Oracle success rate: 252/778 (0.324)
SPL: 151.508/778 (0.195)
Distance to goal: 8.299
Path length: 9.613
```


#### ckpt2000
``` bash
Success rate: 188/778 (0.242)
Oracle success rate: 225/778 (0.289)
SPL: 177.017/778 (0.228)
Distance to goal: 7.910
Path length: 7.573
```

#### ckpt5200
``` bash
Success rate: 233/778 (0.299)
Oracle success rate: 289/778 (0.371)
SPL: 220.358/778 (0.283)
Distance to goal: 7.529
Path length: 8.474
```

#### ckpt7200
``` bash
Success rate: 245/778 (0.315)
Oracle success rate: 286/778 (0.368)
SPL: 232.049/778 (0.298)
Distance to goal: 7.405
Path length: 8.233
```

#### ckpt8200
``` bash
Success rate: 243/778 (0.312)
Oracle success rate: 285/778 (0.366)
SPL: 227.967/778 (0.293)
Distance to goal: 7.146
Path length: 8.198
```

#### ckpt11346
``` bash
Success rate: 240/778 (0.308)
Oracle success rate: 294/778 (0.378)
SPL: 228.662/778 (0.294)
Distance to goal: 7.373
Path length: 8.439
```


### ckpt11346  val_unseen
``` bash
Success rate: 558/1839 (0.303)
Oracle success rate: 654/1839 (0.356)
SPL: 518.712/1839 (0.282)
Distance to goal: 7.991
Path length: 8.802
```

