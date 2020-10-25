import matplotlib.pyplot as plt
import numpy as np
from utils import mismatch, predict_progress
import mpl_toolkits.axisartist as axisartist

plt.style.use('seaborn-white')

# HEK_site1_SpCas9
best_x_3 = [-1.52845086,-0.54307945,3.02779871,1.03082501]
# HEK_site_2_xCas9
best_x_4 = [-4.30187549,-0.5218599,6.3321199,0.49449049]

data = mismatch([[],[1,12]])
energy_3 = predict_progress(data,parms=best_x_3)
energy_4 = predict_progress(data,parms=best_x_4)

# plt.plot(np.arange(23),energy_3[0],'o--',label='HEK_site1_SpCas9_target',c='tab:blue',ms=5)
# plt.plot(np.arange(23),energy_4[0],'o-',label='HEK_site1_xCas9_target',c='tab:red',ms=5)
fig = plt.figure(figsize=(8,4))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
for i in range(23):
    plt.axvline(x=i,c="white",ls="-",lw=1)
ax.plot(np.arange(23),energy_3[1],'o-',label='HEK site1 SpCas9',c='darkcyan')
ax.plot(np.arange(23),energy_4[1],'o-.',label='HEK site1 xCas9',c='tab:purple')

# ax.margins(0) # remove default margins (matplotlib verision 2+)
ax.axvspan(0, 1, facecolor='red', alpha=0.15)
ax.axvspan(21, 22, facecolor='green', alpha=0.15)
ax.axvspan(1, 21, facecolor='gray', alpha=0.15)
ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
ax.axis["left"].set_axisline_style("-|>", size = 1.5)
plt.legend(fontsize=10)
plt.axvline(x=2.5,c="r",ls="-",lw=2)
plt.axvline(x=13.5,c="r",ls="-",lw=2)
plt.xticks([])
plt.xlabel('Targeting Progression')
plt.ylabel(r'Transition-state Free Energy ($k_BT$)')
plt.text(3.2,-3,"mismatch\n position",color='darkblue',size = 10,style = "italic",bbox = dict(facecolor = "b", alpha = 0.2))
plt.text(14.2,-0.5,"mismatch\n position",color='darkblue',size = 10,style = "italic",bbox = dict(facecolor = "b", alpha = 0.2))
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)
plt.savefig('target_process.png',dpi=300)
plt.show()
