# Continual Learning for Blind Image Quality Assessment
The codebase of  [Continual Learning for Blind Image Quality Assessment](https://arxiv.org/abs/2102.09717)

![BIQA_CL_framework](https://user-images.githubusercontent.com/14050646/170612919-af5704c8-c1ec-45c2-89fd-6d71420ca786.png)

# Requirement
torch 1.8+
torchvision
Python 3
sckit-learn
scipy


# Usage
# Replay-free training: 

(1) Using LwF for contiual learning of a model BIQA on six tasks:
```
Modify Line 192 - Line 193 in [BIQA_CL.py]([URL](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py)) to :

method = 'LwF'

training = True

head_usage = 2

Then simply run:

python BIQA_CL.py
```
# Replay-based training: 

(1) Using iCaRL-v2 for contiual learning of a model BIQA on six tasks:

Modify Line 192 - Line 193 in [BIQA_CL.py]([URL](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py)) to :

method = 'LwF-Replay'

training = True

Then simply run:

python BIQA_CL.py

# Inference:

(1) Using the weighted quality predictions for inference:

Modify Line 193 - Line 194 in [BIQA_CL.py]([URL](https://github.com/zwx8981/BIQA_CL/blob/main/BIQA_CL.py)) to :

training = False

head_usage = 2

Then simply run:

python BIQA_CL.py


## Citation

Should you find this repo useful to your research, we sincerely appreciate it if you cite our paper :blush: :
```bash
@article{zhang2022continual,
  title={Continual Learning for Blind Image Quality Assessment},
  author={Zhang, Weixia and Li, Dingquan and Ma, Chao and Zhai, Guangtao and Yang, Xiaokang and Ma, Kede},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={to appear, 2022}
}
```
