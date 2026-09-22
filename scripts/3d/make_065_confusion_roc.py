from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

src=Path('raman_6v4_predictions_all_seeds.csv')
df=pd.read_csv(src)
df['pred_065']=(df['mean_p_cancer']>=0.65).astype(int)
df['correct_065']=(df['pred_065']==df['true']).astype(int)
df.to_csv('raman_6v4_predictions_threshold065.csv',index=False)

seeds=sorted(df.seed.unique())
fig,axs=plt.subplots(1,2,figsize=(10.2,5.1),gridspec_kw={'width_ratios':[1,1.15]})
# Confusion matrix from one OOF score per map for the first sampling seed.
g=df[df.seed==seeds[0]].copy()
cm=confusion_matrix(g.true,g.pred_065,labels=[0,1])
ax=axs[0]
im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(6,cm.max()))
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{cm[i,j]}',ha='center',va='center',fontsize=19,fontweight='bold',color='white' if cm[i,j]>=4 else '#17212b')
ax.set(xticks=[0,1],xticklabels=['Normal','Cancer'],yticks=[0,1],yticklabels=['Normal','Cancer'],xlabel='Predicted class',ylabel='Reference class',title='A  Confusion matrix')
ax.tick_params(length=0)
ax.text(.5,-.19,'Apparent accuracy: 10/10 (100%)',transform=ax.transAxes,ha='center',fontsize=10,fontweight='bold')

# Show all three seed-specific map-level ROC curves; they overlap if rankings match.
ax=axs[1]
colors=['#2b8cbe','#e34a33','#31a354']
aucs=[]
for color,seed in zip(colors,seeds):
    z=df[df.seed==seed]
    fpr,tpr,_=roc_curve(z.true,z.mean_p_cancer)
    auc=roc_auc_score(z.true,z.mean_p_cancer);aucs.append(auc)
    ax.plot(fpr,tpr,lw=2,color=color,label=f'Seed {seed}: AUC={auc:.2f}')
ax.plot([0,1],[0,1],color='#888',ls='--',lw=1,label='Chance')
ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='False-positive rate',ylabel='True-positive rate',title='B  ROC (map-level LOMO)')
ax.grid(color='#e5e7eb',lw=.7);ax.legend(frameon=False,loc='lower right',fontsize=8)
fig.suptitle('Cancer vs Normal classification (6 cancer maps + 4 Normal maps)',fontweight='bold',fontsize=13)
fig.text(.5,.015,'Cutoff 0.65 was selected after reviewing these LOMO results. The 10/10 is post-hoc and is not an independent or unbiased validation accuracy.',ha='center',fontsize=8.5,color='#8b1e1e')
fig.text(.5,.052,'Unit of analysis: map (n=10); confusion matrix shown for seed 123. The three seeds each give 10/10 at 0.65; ROC AUC=1.00 each.',ha='center',fontsize=8.5,color='#333')
fig.tight_layout(rect=[0,.09,1,.90])
for ext in ['png','pdf','svg']:
    fig.savefig(f'raman_6v4_threshold065_confusion_roc.{ext}',dpi=500 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)
# Per-seed summary for traceability.
rows=[]
for seed,z in df.groupby('seed',sort=True):
    c=confusion_matrix(z.true,z.pred_065,labels=[0,1])
    rows.append({'seed':int(seed),'n_maps':len(z),'correct':int(z.correct_065.sum()),'accuracy':float(z.correct_065.mean()),'AUC':float(roc_auc_score(z.true,z.mean_p_cancer)),'TN':int(c[0,0]),'FP':int(c[0,1]),'FN':int(c[1,0]),'TP':int(c[1,1]),'cutoff':0.65})
pd.DataFrame(rows).to_csv('raman_6v4_threshold065_summary.csv',index=False)
