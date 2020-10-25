from utils import predict, mismatch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from itertools import product

pos =  list(product([i for i in range(20)], repeat=2))

# VEGFA_SpCas9
best_x_1 = [-0.48330546,-0.27027048,1.71309145,1.77855314]
# VEGFA_xCas9
best_x_2 = [-1.28784322,-0.45575753,4.94635742,0.09800039]
# HEK_site1_SpCas9
best_x_3 = [-1.52845086,-0.54307945,3.02779871,1.03082501]
# HEK_site_2_xCas9
best_x_4 = [-4.30187549,-0.5218599,6.3321199,0.49449049]

out = mismatch(pos)
pred_1 = predict(out, best_x_1)
pred_2 = predict(out, best_x_2)
pred_3 = predict(out, best_x_3)
pred_4 = predict(out, best_x_4)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
j = 0
for g, pred, title in zip([gs[i] for i in range(4)],
        [pred_1,pred_2,pred_3,pred_4],['VEGFA SpCas9','VEGFA xCas9','HEK site1 SpCas9','HEK site1 xCas9']):
    mat = np.zeros((20,20))
    for i, (index, value) in enumerate(zip(pos,pred)):
        mat[index] = pred[i]
    sns.set()
    ax = plt.subplot(g)
    heatmap = sns.heatmap(mat,ax=ax,cmap="YlGnBu",xticklabels=np.arange(1,21),
                yticklabels=np.arange(1,21),cbar=j == 0,cbar_ax=None if j else cbar_ax,
                cbar_kws=None if j else {"label":"Off-target Probability"},
                vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    j += 1


fig.text(0.5, 0, 'Mismatch Position', ha='center')
fig.text(0, 0.5, 'Mismatch Position', va='center', rotation='vertical')
fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('multi-mismatch.png',dpi=300)
plt.show()