#!/usr/bin/env python3
"""Run map-level LOMO Raman classification on the ten currently uploaded maps.

Current upload contains 6 cancer maps + 4 Normal maps; no Pneumonia map.
This script reports map-level leave-one-map-out results, not patient-level validation.
"""
from pathlib import Path
import sys, csv
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve)

CANCER = ["FZXT_layer8(1).mat", "Micropapillary_layer8(2).mat",
          "Papillary_layer8(2).mat", "Solid_layer8(2).mat",
          "TieBi_layer8(2).mat", "XianPao_layer8(2).mat"]
NORMAL = ["Normal-ChenRongMei_layer1(1).mat", "Normal-CuiDaoFen_layer1(1).mat",
          "Normal-HuangWenXiang_layer1(1).mat", "Normal-ZhangLiYing_layer1(1).mat"]
SEEDS = [123, 321, 2026]


def load_map(path, rng):
    st = scipy.io.loadmat(path)["data_struct"][0, 0]
    raw = np.asarray(st["data"], dtype=np.float32)
    axis = np.asarray(st["axisscale"][1, 0][0], dtype=np.float32).ravel()
    n = min(1800, raw.shape[0])
    raw = raw[rng.choice(raw.shape[0], n, replace=False)]
    wave = np.arange(400.0, 3200.0, 5.0)
    keep = (((wave >= 400) & (wave <= 1800)) |
            ((wave >= 2800) & (wave <= 3100)))
    wave = wave[keep]
    x = np.vstack([np.interp(wave, axis, r) for r in raw]).astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    x /= np.std(x, axis=1, keepdims=True) + 1e-6
    return x


def evaluate(data_dir, seed):
    rng = np.random.default_rng(seed)
    specs = [(name, 1) for name in CANCER] + [(name, 0) for name in NORMAL]
    maps = [(name, load_map(data_dir / name, rng), y) for name, y in specs]
    results = []
    for i, (name, xtest, ytest) in enumerate(maps):
        train = [m for j, m in enumerate(maps) if i != j]
        xtrain = np.concatenate([m[1] for m in train])
        ytrain = np.concatenate([np.full(len(m[1]), m[2], np.int8) for m in train])
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
            l2_regularization=1, learning_rate=0.08, random_state=42)
        model.fit(xtrain, ytrain)
        score = float(model.predict_proba(xtest)[:, 1].mean())
        pred = int(score >= 0.5)
        results.append(dict(seed=seed, map=name, true=ytest,
            mean_p_cancer=score, threshold=0.5, pred=pred, correct=int(pred == ytest)))
    return results


def figure(results, outdir):
    y = np.array([r["true"] for r in results]); pred = np.array([r["pred"] for r in results])
    scores = np.array([r["mean_p_cancer"] for r in results])
    cm = confusion_matrix(y, pred, labels=[0,1]); auc = roc_auc_score(y, scores)
    names = [r["map"].replace("_layer1(1).mat", "").replace("_layer8(1).mat", "").replace("_layer8(2).mat", "") for r in results]
    colors = ["#C65353" if v else "#3977A8" for v in y]
    fig, axs = plt.subplots(1,3,figsize=(13,4.8),gridspec_kw={"width_ratios":[1.45,1,1]})
    ax=axs[0]; yy=np.arange(len(y))[::-1]
    for yi,s,c,nm in zip(yy,scores,colors,names):
        ax.plot([0,s],[yi,yi],color=c,alpha=.23,lw=1.2)
        ax.scatter(s,yi,color=c,s=46,edgecolor='white',linewidth=.5,zorder=2)
        ax.text(min(s+.02,.94),yi,f'{s:.2f}',va='center',fontsize=8)
    ax.axvline(.5,color='#333',ls='--',lw=1)
    ax.set(yticks=yy,yticklabels=names,xlim=(0,1.05),xlabel='Mean map-level P(cancer)',title='A  Held-out map scores')
    ax.grid(axis='x',color='#e5e7eb',lw=.6); ax.set_axisbelow(True); ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y',length=0)
    ax=axs[1]; ax.imshow(cm,cmap='Blues',vmin=0,vmax=6,aspect='auto')
    for i in range(2):
        tot=cm[i].sum()
        for j in range(2):
            pct=100*cm[i,j]/tot if tot else 0
            ax.text(j,i,f'{cm[i,j]}\n({pct:.0f}%)',ha='center',va='center',fontweight='bold',color='white' if cm[i,j]>=4 else '#222')
    ax.set(xticks=[0,1],xticklabels=['Normal','Cancer'],yticks=[0,1],yticklabels=['Normal','Cancer'],xlabel='Predicted',ylabel='Reference',title=f'B  Accuracy {accuracy_score(y,pred):.1%}')
    ax.tick_params(length=0)
    ax=axs[2]; fpr,tpr,_=roc_curve(y,scores)
    ax.plot(fpr,tpr,color='#3B7A78',lw=2,label=f'LOMO AUC={auc:.2f}');ax.plot([0,1],[0,1],color='#888',ls='--',lw=1,label='Chance')
    ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='False-positive rate',ylabel='True-positive rate',title='C  ROC (n=10 maps)')
    ax.grid(color='#e5e7eb',lw=.6);ax.legend(frameon=False,loc='lower right',fontsize=8)
    fig.suptitle('Exploratory Raman cancer vs Normal classification',fontweight='bold')
    fig.text(.01,-.015,'6 cancer maps + 4 Normal maps; leave-one-map-out. Fixed cutoff=0.5. AUC is not accuracy; patient independence not verified.',fontsize=7.5,color='#444')
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(outdir/f'raman_6v4_lomo.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)


def main():
    data_dir=Path(sys.argv[1]) if len(sys.argv)>1 else Path('upload')
    missing=[f for f in CANCER+NORMAL if not (data_dir/f).exists()]
    if missing: raise SystemExit('Missing input files:\n'+'\n'.join(missing))
    allr=[]
    for seed in SEEDS:
        r=evaluate(data_dir,seed);allr+=r
        y=np.array([x['true'] for x in r]);p=np.array([x['pred'] for x in r]);s=np.array([x['mean_p_cancer'] for x in r])
        print(f'seed={seed}: {int((y==p).sum())}/10 correct; accuracy={accuracy_score(y,p):.4f}; balanced_accuracy={balanced_accuracy_score(y,p):.4f}; AUC={roc_auc_score(y,s):.4f}; confusion={confusion_matrix(y,p,labels=[0,1]).tolist()}')
        for x in r: print(x)
        if seed==SEEDS[0]:
            figure(r,Path.cwd())
            with open('raman_6v4_predictions_seed123.csv','w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
    with open('raman_6v4_predictions_all_seeds.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(allr[0]));w.writeheader();w.writerows(allr)

if __name__=='__main__': main()
