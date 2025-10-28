import numpy as np
import torch
import torch.nn.functional as F


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)
    if (x.size(1)==batch_size):
        logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)
    else:
        logsumexp = torch.logsumexp(x,dim=dim)

    num_elem = torch.count_nonzero(x).item()

    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba_lower_bound(scores, mask,log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]

    p_scores = scores * mask
    n_scores = scores * (1-mask)
    p_sum = torch.sum(p_scores)
    joint_term = p_sum/torch.count_nonzero(p_scores).item()
    marg_term = logmeanexp_nodiag(n_scores).exp()
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores,mask):
    return tuba_lower_bound(scores - 1.,mask)


def js_fgan_lower_bound(f,mask):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    q_maxtrix = f * mask
    non_index = q_maxtrix.nonzero()
    nonzero_elements = q_maxtrix[non_index[:, 0],non_index[:, 1]].flatten()

    # f_diag = f.diag()
    first_term = -F.softplus(-nonzero_elements).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(nonzero_elements))) / (n * (n - 1.))
    return first_term - second_term


def js_lower_bound(f,mask):
    """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
    nwj = nwj_lower_bound(f,mask)
    js = js_fgan_lower_bound(f,mask)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def estimate_mutual_information(x, y, critic_fn,mask):

    x,y  = x.cuda(),y.cuda()

    scores = critic_fn(x, y)

    mi = js_lower_bound(scores,mask)

    return mi
