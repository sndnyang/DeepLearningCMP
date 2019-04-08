import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(d):
    d = d.cpu().numpy()
    axis = tuple(range(1, len(d.shape)))
    reshape = tuple([-1] + [1] * (len(d.shape) -1))
    d /= (np.sqrt(np.sum(d ** 2, axis=axis)).reshape(
        reshape) + 1e-16)
    return torch.from_numpy(d)


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))


class VAT(object):

    def __init__(self, device, eps, xi, k=1, use_entmin=False, debug=False):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.k = k
        self.debug = debug
        try:
            self.kl_div = nn.KLDivLoss(reduction='none')
        except TypeError:
            self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_entmin = use_entmin

    def __call__(self, model, X):
        logits = model(X, update_batch_stats=False)
        prob_logits = F.softmax(logits.detach(), dim=1)
        d = VAT.approx_power_iter(model, X, prob_logits, self.xi, self.k, self.device, self.debug)
        logits_hat = model(X + self.eps * d, update_batch_stats=False)
        LDS = torch.mean(self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
        # LDS = ((F.softmax(logits_hat, dim=1) - prob_logits) ** 2).sum(dim=1).mean()
        if self.debug:
            print("LDS value", LDS.item())

        if self.use_entmin:
            LDS += _entropy(logits_hat)

        return LDS

    @staticmethod
    def approx_power_iter(model, X, prob_logits, xi, k, device, debug=False):
        try:
            kl_div = nn.KLDivLoss(reduction='none')
        except TypeError:
            kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        d = _l2_normalize(torch.randn(X.size())).to(device)
        if debug:
            print("random init d", d.sum())

        for ip in range(k):
            X_hat = X + d * xi
            X_hat.requires_grad = True
            logits_hat = model(X_hat, update_batch_stats=False)
            if debug:
                print("logit output", (logits_hat ** 2).sum())
                prob_hat = F.softmax(logits_hat, dim=1)
                print("pro output", (prob_logits ** 2).sum())
                print("pro hat output", (prob_hat ** 2).sum())

            adv_distance = torch.mean(kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
            # adv_distance = ((F.softmax(logits_hat, dim=1) - prob_logits) ** 2).sum()
            if debug:
                print("distance", adv_distance.item())
            adv_distance.backward()
            grad_x_hat = X_hat.grad
            # grad_x_hat = torch.autograd.grad(adv_distance, X_hat)[0]
            if debug:
                print("grad on x_hat", grad_x_hat.sum())
            d = _l2_normalize(grad_x_hat).to(device)
        return d
