import argparse
import torch

def top_k_error(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = torch.topk(input=output, k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.flatten(correct[:k]).float().sum(0).cpu().numpy().item()
        res.append((1.0-correct_k/batch_size)*100.0)
    return res


def run(model, criterion, dataset, device):
    with torch.no_grad():
        model.eval()
        # TODO