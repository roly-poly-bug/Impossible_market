from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
from ml.experiments.mf_bias_experiment import *
from ml.training.mf_trainer import save_checkpoint

H=("epoch","train_loss","validation_purchase_recall_at_10","validation_purchase_ndcg_at_10","validation_viewplus_ndcg_at_10","validation_favoriteplus_ndcg_at_10")
M=("model","task","split","k","eligible_users","recall","ndcg","hit_rate","precision")
C=("model","k","eligible_users","recall","ndcg","hit_rate","precision")
B=("model","component","mean","std","min","median","max")
P=("model","unique_purchase_top10_lists","average_pairwise_top10_overlap","average_cart_popularity_top10_overlap","recommended_item_cart_score_mean")
def _csv(path,rows,cols):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator="\n");w.writeheader();n=0
        for row in rows:w.writerow({c:row[c] for c in cols});n+=1
    return n
def _json(path,v):path.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def _meta(path,n=None):
    v={"bytes":path.stat().st_size,"sha256":_sha(path)}
    if n is not None:v["data_rows"]=n
    return v
def export_bias_results(result,output_dir,*,dataset_dir,popularity_dir,negative_results_dir):
    d=Path(output_dir);d.mkdir(parents=True,exist_ok=True);cp=d/"checkpoints";cp.mkdir(exist_ok=True);manifests={}
    for name in MODELS:
        md=d/name;md.mkdir(exist_ok=True);hp=md/"training_history.csv";mp=md/"metrics.csv"
        hn=_csv(hp,result.histories[name],H);mn=_csv(mp,({"model":name,**r} for r in result.metrics[name]),M)
        dp=md/"diagnostics.json";_json(dp,result.diagnostics[name]);cfg=md/"config.json"
        _json(cfg,{"experiment_version":EXPERIMENT_VERSION,"model":name,"global_bias":False,"bias_initialization":"zero","bias_regularization":"same Adam weight_decay as embeddings","fixed_parameters":result.config.as_dict(),"positive_confidence":"Weighted Implicit v1 unchanged","sampling":"Exposed Non-conversion with Unknown backfill; shared samples","selection_metric":"validation_purchase_ndcg_at_10","test_policy":"one final Test; no-bias reused"})
        artifacts={hp.name:_meta(hp,hn),mp.name:_meta(mp,mn),dp.name:_meta(dp),cfg.name:_meta(cfg)}
        if name in TRAINED_MODELS:
            p=cp/f"{name}_best.pt";save_checkpoint(result.training[name],p);artifacts[f"../checkpoints/{p.name}"]=_meta(p);be=result.training[name].best_epoch;bn=result.training[name].best_validation_purchase_ndcg_at_10
        else:be=24;bn=max(float(r["validation_purchase_ndcg_at_10"]) for r in result.histories[name])
        mf={"model":name,"best_epoch":be,"best_validation_purchase_ndcg_at_10":bn,"artifacts":artifacts};mfp=md/"manifest.json";_json(mfp,mf);manifests[name]=_meta(mfp)
    tables={"comparison.csv":(result.comparison,C),"bias_diagnostics.csv":(result.bias_rows,B),"personalization.csv":(result.personalization,P)};arts={}
    for fn,(rows,cols) in tables.items():p=d/fn;arts[fn]=_meta(p,_csv(p,rows,cols))
    root={"experiment_version":EXPERIMENT_VERSION,"models":list(MODELS),"fixed_mf_config":result.config.as_dict(),"shared_sample_count":len(result.data.sampled.triples),"global_bias":False,"test_policy":"frozen no-bias reused; two bias models tested once; no tuning","dataset_manifest_sha256":_sha(Path(dataset_dir)/"manifest.json"),"negative_results_manifest_sha256":_sha(Path(negative_results_dir)/"manifest.json"),"model_manifests":manifests,"artifacts":arts};_json(d/"manifest.json",root);return root
