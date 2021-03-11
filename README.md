# BIQA_CL
The codebase for  [Continual Learning for Blind Image Quality Assessment](https://arxiv.org/abs/2102.09717)

# Usage of MPSR.py:

Given a SRCC_Maxtrix, whose shape is  T * T, where T is the length of a task sequence, SRCC_Maxtrix(t, s) is the SRCC result of the s-th dataset when the model is trained on the t-th dataset.

import CL_Metric

mpsr, psrt = CL_Metric(SRCC_Matrix)


Training code is coming soon!

