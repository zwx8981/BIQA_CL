# Continual Learning for Blind Image Quality Assessment
The codebase of  [Continual Learning for Blind Image Quality Assessment](https://arxiv.org/abs/2102.09717)

![BIQA_CL_framework](https://user-images.githubusercontent.com/14050646/170612919-af5704c8-c1ec-45c2-89fd-6d71420ca786.png)

# Requirement
torch 1.8+
torchvision
Python 3
scikit-learn
scipy


# Usage
# Replay-free training: 

(1) Using LwF for continual learning of a model BIQA on six tasks:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :
```
method = 'LwF'  
training = True  
```

```
Then run in terminal: python BIQA_CL.py
```

(2) Using other continual learning methods:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :
```
method = 'EWC' /  'SI' / 'MAS'   
training = True  
```
Set appropriate regularization weight by modifying Line 84 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py):

```
1000 for si, 10 for mas, 10000 for ewc
```

```
Then run in terminal: python BIQA_CL.py
```

(3) Baselines:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :
```
method = 'SL'  / 'SH-CL' / 'MH-CL'
training = True  
```

```
Then run in terminal: python BIQA_CL.py
```

# Replay-based training: 

(1) Using iCaRL for continual learning of a model BIQA on six tasks:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
method = 'LwF-Replay'
training = True
```

Set Line 98 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py)

```
new_replay = False %for iCaRL-v1  
or  
new_replay = True %for iCaRL-v2
```

```
Then run in terminal: python BIQA_CL.py
```

(2) Using other replay methods:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
method = 'SH-CL-Replay' / 'MH-CL-Replay'
training = True
```

```
Then run in terminal: python BIQA_CL.py
```

# Joint Learning:

Modify Line 192 - Line 193 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
method = 'JL'
training = True
```

```
Then run in terminal: python BIQA_CL.py
```

# Inference:

(1) Using the weighted quality predictions for inference, can be used with models trained by LwF / Reg-CL / MH-CL / MH-CL-Replay / iCaRL-v1 / iCaRL-v2:

Modify Line 193 - Line 194 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
training = False
head_usage = 2
```

```
Then run in terminal: python BIQA_CL.py
```

(2) Using task-oracle information for inference, can be used with models trained by  LwF / Reg-CL / MH-CL / MH-CL-Replay / iCaRL-v1 / iCaRL-v2:

Modify Line 193 - Line 194 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
training = False
head_usage = 1
```

```
Then run in terminal: python BIQA_CL.py
```

(3) Using the prediction head trained in the latest task for inference, can be used with models trained by  LwF / Reg-CL / MH-CL / MH-CL-Replay / iCaRL-v1 / iCaRL-v2:

Modify Line 193 - Line 194 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
training = False
head_usage = 0
```

```
Then run in terminal: python BIQA_CL.py
```

(4) Using the single head for inference, can be used with models trained by  SL / SH-CL / SH-CL-Replay / JL:

Modify Line 193 - Line 194 in [BIQA_CL.py](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py) to :

```
training = False
head_usage = 3
```

```
Then run in terminal: python BIQA_CL.py
```


## Citation

Should you find this repo useful to your research, we sincerely appreciate it if you cite our paper :blush: :
```bash
@article{zhang2023continual,
  title={Continual Learning for Blind Image Quality Assessment},
  author={Zhang, Weixia and Li, Dingquan and Ma, Chao and Zhai, Guangtao and Yang, Xiaokang and Ma, Kede},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  month={Mar.},
  volume={45},
  issue={3},
  pages={2864 - 2878},
  year={2023}
}
```
