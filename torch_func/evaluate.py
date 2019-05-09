import torch
import torch.nn as nn
import torch.nn.functional as nfunc


def evaluate_classifier(classifier, loader, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    n_err = 0
    t_loss = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            logits = classifier(x.to(device))
            loss = criterion(logits, y).item()
            t_loss += loss * y.shape[0]
            prob_y = nfunc.softmax(logits, dim=1)
            pred_y = torch.max(prob_y, dim=1)[1]
            n_err += torch.sum(pred_y != y).item()

    classifier.train()

    return n_err, t_loss / len(loader.dataset)


def mse(p, q):
    criterion = nn.MSELoss()
    with torch.no_grad():
        loss = criterion(nfunc.softmax(p, dim=1), nfunc.softmax(q, dim=1)).item()
    return loss


def kl_div(p, q):
    criterion = nn.KLDivLoss(reduction="batchmean")
    with torch.no_grad():
        loss = criterion(nfunc.log_softmax(p, dim=1), nfunc.softmax(q, dim=1)).item()
    return loss
