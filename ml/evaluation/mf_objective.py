from __future__ import annotations

from statistics import fmean, median, pstdev

import torch


def pairwise_margin_diagnostics(model, triples):
    with torch.no_grad():
        users = torch.tensor([row[0] for row in triples])
        positives = torch.tensor([row[1] for row in triples])
        comparisons = torch.tensor([row[2] for row in triples])
        values = (model(users, positives) - model(users, comparisons)).cpu().tolist()
    return {"count": len(values), "mean": fmean(values), "std": pstdev(values), "min": min(values), "median": median(values), "max": max(values), "positive_share": sum(value > 0 for value in values)/len(values)}
