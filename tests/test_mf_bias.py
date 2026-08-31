from __future__ import annotations
from pathlib import Path
import torch
from ml.models.matrix_factorization import MatrixFactorization
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.evaluation.mf_bias import _correlations
from ml.experiments.mf_bias_export import _csv

def test_no_bias_score_correctness():
    m=MatrixFactorization(1,2,2)
    with torch.no_grad():m.user_embeddings.weight[:]=torch.tensor([[2.,3.]]);m.item_embeddings.weight[:]=torch.tensor([[4.,5.],[1.,1.]])
    assert m(torch.tensor([0]),torch.tensor([0])).item()==23

def test_item_bias_score_correctness_and_zero_initialization():
    m=BiasedMatrixFactorization(1,2,2,"item_bias")
    assert torch.equal(m.item_bias.weight,torch.zeros_like(m.item_bias.weight))
    with torch.no_grad():m.user_embeddings.weight[:]=torch.tensor([[1.,2.]]);m.item_embeddings.weight[:]=torch.tensor([[3.,4.],[1.,1.]]);m.item_bias.weight[:]=torch.tensor([[.5],[-.5]])
    assert m(torch.tensor([0]),torch.tensor([0])).item()==11.5

def test_user_item_bias_score_and_ranking_invariance():
    m=BiasedMatrixFactorization(1,2,2,"user_item_bias")
    assert torch.equal(m.user_bias.weight,torch.zeros_like(m.user_bias.weight))
    with torch.no_grad():m.user_embeddings.weight[:]=torch.tensor([[1.,0.]]);m.item_embeddings.weight[:]=torch.tensor([[2.,0.],[1.,0.]]);m.item_bias.weight.zero_();m.user_bias.weight[:]=3
    assert m(torch.tensor([0]),torch.tensor([0])).item()==5
    assert torch.argsort(m.score_all_items(torch.tensor([0]))[0],descending=True).tolist()==[0,1]

def test_bias_parameters_train_and_receive_regularization():
    m=BiasedMatrixFactorization(1,2,2,"user_item_bias");opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-4)
    before=m.item_bias.weight.detach().clone();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(torch.tensor([0,0]),torch.tensor([0,1])),torch.tensor([1.,0.]));loss.backward();opt.step()
    assert not torch.equal(before,m.item_bias.weight)
    assert m.user_bias.weight.grad is not None

def test_bias_correlations_diagnostics():
    value=_correlations([1.,2.,3.],[2.,4.,6.]);assert value["pearson"]==1;assert value["spearman"]==1

def test_artifact_reproducibility(tmp_path:Path):
    rows=[{"model":"item_bias","value":1}];a=tmp_path/"a.csv";b=tmp_path/"b.csv";_csv(a,rows,("model","value"));_csv(b,rows,("model","value"));assert a.read_bytes()==b.read_bytes()

def test_bias_source_has_no_future_leakage():
    source=Path("ml/experiments/mf_bias_experiment.py").read_text(encoding="utf-8")
    assert "validation_relevance" not in source and "test_relevance" not in source
