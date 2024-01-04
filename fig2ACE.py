import numpy as np
import matplotlib.pyplot as plt


alt_color = '#e07a5f'

##
## Panel A
##
fs_data_WT = np.array([7.488210756540372,	14.211122712045562,	24.905018036429137,	34.597186416020506,	44.99893612283577,	54.68754175099336,	64.40464939062106,	74.3007852897952,	84.06242732234489])
taus_data_WT = np.array([0.00654804,	0.015326264,	0.021569986,	0.060916617,	0.110128111,	0.152437763	,0.171043243,	0.140756887,	0.122324596])
taus_plus_error_WT =np.array([0.006353476	,0.018999163	,0.026084813	,0.068827327	,0.127659733	,0.173202439,	0.198462435,	0.16498715,	0.149683223])
fs_data_N138G = [7.179487179487186,	13.846153846153854,	26.66666666666667,	35.72649572649573,	44.44444444444444,	54.871794871794876,	64.27350427350426,	74.70085470085469]
taus_data_N138G = [0.003508772,	0.012631579	,0.092631579	,0.130526316	,0.224561404	,0.190877193,	0.169122807	,0.133333333]
taus_plus_error_N138G = np.array([0.007607557,	0.015910492	,0.114761187,	0.144640164	,0.247007868,	0.209981947	,0.18829298	,0.160333722])


WT_file = np.load('L-PSGL1.npz')
fs_WT = WT_file['fs']
taus_WT = WT_file['taus']

N138G_file = np.load('LN138G.npz')
fs_N138G = N138G_file['fs']
taus_N138G = N138G_file['taus']

plt.figure(figsize=(3,2))

plt.plot(fs_N138G,taus_N138G,color=alt_color)
plt.plot(fs_data_N138G,taus_data_N138G,'s',color=alt_color,markersize=5)
plt.errorbar(fs_data_N138G,taus_data_N138G,yerr=np.abs(taus_plus_error_N138G - taus_data_N138G),fmt='None',capsize=2,color=alt_color)

plt.plot(fs_WT,taus_WT,color='k')
plt.plot(fs_data_WT,taus_data_WT,'D',color='k',markersize=4)
plt.errorbar(fs_data_WT,taus_data_WT,yerr=np.abs(taus_plus_error_WT-taus_data_WT),fmt='None',capsize=2,color='k')

plt.xlim(0,100)
plt.ylim(0,.26)
plt.gca().set_xticks([0,50,100])
plt.gca().set_xticklabels([0,'',100])
plt.gca().set_yticks([0,0.1,0.2])
plt.gca().set_yticklabels([0,'',0.2])




##
## Panel C
##
fs_data_WT = np.array([12.0,	16.846153846153847,	22.15384615384616,	28.846153846153847,	36.92307692307693,	43.38461538461538,	55.15384615384617,	78,	103.84615384615384])
taus_data_WT = np.array([0.026634383,	0.037530266	,0.064164649,	0.095641646	,0.163438257,	0.129539952	,0.071428571,	0.049636804	,0.041162228])
taus_plus_error_WT = np.array([0.034504792	,0.051118211	,0.081789137,	0.115015974	,0.200638978	,0.157188498	,0.086900958	,0.063897764	,0.052396166])
fs_data_A108H = np.array([7.831094049904031,	12.898272552783112,	19.577735124760082,	29.251439539347423,	36.852207293666034,	44.68330134357007,	54.12667946257198,	66.10364683301344,	78.0806142034549,	94.43378119001923])
taus_data_A108H = np.array([0.3284236,	0.298179365,	0.252218421,	0.242459225,	0.181996254	,0.131192801	,0.119020464	,0.114069005	,0.071677933,	0.02078871])
taus_plus_error_A108H = np.array([0.383386581,	0.350159744	,0.277316294,	0.284984026,	0.222364217,	0.159744409	,0.152076677	,0.146964856,	0.089456869	,0.02172524])

WT_file = np.load('L-2GSP6.npz')
fs_WT = WT_file['fs']
taus_WT = WT_file['taus']

N138G_file = np.load('L-A108H.npz')
fs_A108H = N138G_file['fs']
taus_A108H = N138G_file['taus']

plt.figure(figsize=(3,2))

plt.plot(fs_A108H,taus_A108H,color=alt_color)
plt.plot(fs_data_A108H,taus_data_A108H,'s',color=alt_color,markersize=5)
plt.errorbar(fs_data_A108H,taus_data_A108H,yerr=np.abs(taus_plus_error_A108H - taus_data_A108H),fmt='None',capsize=2,color=alt_color)

plt.plot(fs_WT,taus_WT,color='k')
plt.plot(fs_data_WT,taus_data_WT,'D',color='k',markersize=4)
plt.errorbar(fs_data_WT,taus_data_WT,yerr=np.abs(taus_plus_error_WT-taus_data_WT),fmt='None',capsize=2,color='k')

plt.xlim(0,120)
plt.ylim(0,.4)
plt.gca().set_xticks([0,50,100])
plt.gca().set_xticklabels([0,'',100])
plt.gca().set_yticks([0,0.2,0.4])
plt.gca().set_yticklabels([0,'',0.4])



##
## Panel E
##
fs_data = np.array([4.75297619047619,	6.74867724867725,	8.73875661375661,	10.92691798941799,	14.070767195767196,	17.611441798941797,	26.80820105820106,	36.088955026455025])
taus_data = np.array([0.106666667,	0.265,	0.395,	0.623333333,	0.268333333,	0.113333333,	0.115,	0.09])
SDs = np.array([0.103892613,	0.335503363,	0.405707018	,0.649527099,	0.44570724,	0.299775074,	0.233476805	,0.112356505])
invSlopes = np.array([0.125363825,	0.391060291	,0.454677755,	0.703534304	,0.361122661,	0.052390852,	0.084199584, np.nan])

file = np.load('P-sPSGL1.npz')
fs = file['fs']
taus = file['taus']

plt.figure(figsize=(3,2))

plt.plot(fs,taus,color='k')
plt.plot(fs_data,taus_data,'D',color='k',markersize=5)
plt.plot(fs_data,SDs,'^',color='#d90429',markersize=4)
plt.plot(fs_data,invSlopes,'s',color='#118ab2',markersize=3)

plt.xlim(0,50)
plt.ylim(0,.75)
plt.gca().set_xticks([0,25,50])
plt.gca().set_xticklabels([0,'',50])
plt.gca().set_yticks([0,0.2,0.4,.6])
plt.gca().set_yticklabels([0,'','',.6])



plt.show()

