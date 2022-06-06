import torch
import itertools
from random import sample
EPS = 1e-2
eps = 1e-8

class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))

        return torch.mean(loss)

class Focal_Fidelity_Loss(torch.nn.Module):

    def __init__(self, gamma=1):
        super(Focal_Fidelity_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, p, g, alpha=1):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        #loss = alpha * (1 - torch.exp(-self.gamma*torch.abs(p-g))*(torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps)))

        fidelity_loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        focal_fidelity_loss = alpha * (1 - torch.exp(-self.gamma*fidelity_loss)*(torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps)))

        return torch.mean(focal_fidelity_loss)


class Sigma_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Sigma_Fidelity_Loss, self).__init__()

    def forward(self, p, g, sigma1, sigma2):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
        #loss = 0.5 * loss / (sigma1*sigma1 + sigma2*sigma2 + eps)
        #loss += 0.5 * torch.log(sigma1*sigma1 + sigma2*sigma2 + eps)

        loss = 0.5 * loss / (sigma1 * sigma1 + sigma2 * sigma2 + eps)
        loss += 0.5 * (sigma1 * sigma1 + sigma2 * sigma2 + eps)

        return torch.mean(loss)

class Pairwise_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Pairwise_Fidelity_Loss, self).__init__()

    def forward(self, pmos, gmos, gstd, pstd=None, ratio=1):
        loss = []
        pairs = []
        combs = itertools.combinations(range(0, pmos.size(0)), 2)
        for pair in combs:
            pairs.append(pair)

        pairs = sample(pairs, int(ratio*len(pairs)))

        for pair in pairs:
            idx1 = pair[0]
            idx2 = pair[1]

            if pstd is None:
                constant = torch.sqrt(torch.Tensor([2])).to(pmos.device)
                p = 0.5 * (1 + torch.erf((pmos[idx1] - pmos[idx2]) / constant))
            else:
                p_var = pstd[idx1] * pstd[idx1] + pstd[idx1] * pstd[idx2] + eps
                p = 0.5 * (1 + torch.erf((pmos[idx1] - pmos[idx2]) / torch.sqrt(p_var)))

            g_var = gstd[idx1] * gstd[idx1] + gstd[idx1] * gstd[idx2] + eps
            g = 0.5 * (1 + torch.erf((gmos[idx1] - gmos[idx2]) / torch.sqrt(g_var)))

            g = g.view(-1, 1)
            p = p.view(-1, 1)
            #loss += 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
            loss_item = 1 - (torch.sqrt(p * g + eps) + torch.sqrt((1 - p) * (1 - g) + eps))
            loss.append(loss_item)

        return loss
