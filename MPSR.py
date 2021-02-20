import numpy as np

def CL_Metric(srcc_matrix):
    T = np.shape(srcc_matrix)[0]
    MPSR = 0
    PSR_t = []
    for t in range(0, T):
        plasticity = srcc_matrix[t, t]
        PSR = 0
        if t == 0:
            PSR = plasticity
        else:
            for s in range(0, t):
                PSR += srcc_matrix[t, s] / srcc_matrix[s, s]
            PSR = PSR * (1 / t) * plasticity
        PSR_t.append(PSR)
        MPSR += PSR
    MPSR = MPSR / T
    return MPSR, PSR_t

