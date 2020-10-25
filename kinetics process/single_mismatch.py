from utils import predict, get_corrcoef
import numpy as np
import matplotlib.pyplot as plt

for i in range(20):
    temp = np.zeros((2,20))
    temp[0,:] = 1
    temp[1,i] = 1
    temp[0,i] = 0
    temp = temp[np.newaxis,:]        
    out = temp if i == 0 else np.concatenate((out,temp),axis=0)

# vcf_2
# best_x = [-1.14239202,-1.06746578,7.27654574,6.02079389]
# VEGFA_SpCas9
best_x_1 = [-0.48330546,-0.27027048,1.71309145,1.77855314]
# VEGFA_xCas9
best_x_2 = [-1.28784322,-0.45575753,4.94635742,0.09800039]
# HEK_site1_SpCas9
best_x_3 = [-1.52845086,-0.54307945,3.02779871,1.03082501]
# HEK_site_2_xCas9
best_x_4 = [-4.30187549,-0.5218599,6.3321199,0.49449049]
pred_1 = predict(out, best_x_1)
pred_2 = predict(out, best_x_2)
pred_3 = predict(out, best_x_3)
pred_4 = predict(out, best_x_4)
plt.plot(np.arange(1,21),pred_1[::-1],'o--',label='VEGFA_SpCas9',c='tab:blue',ms=5)
plt.plot(np.arange(1,21),pred_2[::-1],'o-',label='VEGFA_xCas9',c='tab:red',ms=5)
plt.plot(np.arange(1,21),pred_3[::-1],'^--',label='HEK_site1_SpCas9',c='darkcyan')
plt.plot(np.arange(1,21),pred_4[::-1],'^-',label='HEK_site1_xCas9',c='tab:purple')
plt.xticks(np.arange(1,21))
plt.title("Single Mismatch",size=16)
plt.xlabel("Mismatch Position")
plt.ylabel("Off-Target Probability")
# plt.axvline(x=8,ymin=0,ymax=1,linestyle=':',c='tab:gray',linewidth=2)
# plt.text(4,0.05,"Seed Region",ha='center',color='tab:gray')
plt.legend()

plt.savefig("single_mismatch.svg")
plt.savefig("single_mismatch.png",dpi=300)
plt.show()

for data,parms in zip(['VEGFA_SpCas9.csv','VEGFA_xCas9.csv','HEK_site1_SpCas9.csv','HEK_site1_xCas9.csv'],
                        [best_x_1, best_x_2, best_x_3, best_x_4]):
    get_corrcoef(data, parms)

