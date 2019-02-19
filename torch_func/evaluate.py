import torch
import torch.nn as nn
import torch.nn.functional as F


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
            logits = classifier(x.to(device))
            loss = criterion(logits, y.to(device)).item()
            t_loss += loss * y.shape[0]
            prob_y = F.softmax(logits, dim=1)
            pred_y = torch.max(prob_y, dim=1)[1]
            pred_y = pred_y.to(torch.device('cpu'))
            n_err += (pred_y != y).sum().item()

    classifier.train()

    return n_err, t_loss / len(loader.dataset)
