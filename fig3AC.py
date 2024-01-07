import numpy as np
import matplotlib.pyplot as plt


alt_color = '#e07a5f'


##
## Panel A
##

fs_data= np.array([12.838095238095235,	23.352380952380948,	27.61904761904762,	32.34285714285715,	35.39047619047619,	38.43809523809524,	43.161904761904765,	48.49523809523809,	51.542857142857144,	56.266666666666666,	65.10476190476192,	72.87619047619047,	78.20952380952382,	83.6952380952381,	97.86666666666667])
taus_data = np.array([0.46367713,0.443497758,0.423318386,0.401793722,0.404484305	,0.452914798,0.471748879,	0.455605381,	0.347982063,	0.306278027,	0.225560538,	0.186547085	,0.187892377	,0.165022422	,0.109865471]) # s
SDs = np.array([0.49478673,	0.411374408,	0.4	,0.35450237	,0.381042654,	0.407582938	,0.636966825,	0.472037915,	0.411374408	,0.301421801,	0.263507109	,0.225592417,	0.193364929	,0.172511848,	0.106161137])
invSlopes = np.array([0.508056872	,0.437914692,	0.426540284,	0.436018957,	0.420853081	,0.447393365,	0.678672986,	0.51563981,	0.447393365	,0.312796209	,0.276777251	,0.244549763	,0.206635071	,0.172511848	,0.111848341])

file = np.load('E_sLex.npz')
fs = file['fs']
taus = file['taus']

plt.figure(figsize=(3,2))


plt.plot(fs_data,taus_data,'D',color='k',markersize=5)
plt.plot(fs_data,SDs,'^',color='#d90429',markersize=4)
plt.plot(fs_data,invSlopes,'s',color='#118ab2',markersize=3)
plt.plot(fs,taus,color='k')

plt.xlim(0,110)
plt.ylim(0,.7)
plt.gca().set_xticks([0,50,100])
plt.gca().set_xticklabels([0,'',100])
plt.gca().set_yticks([0,0.2,0.4,.6])
plt.gca().set_yticklabels([0,'','',.6])



##
## Panel C
##
fs_data= np.array([12.838095238095235,	23.352380952380948,	27.61904761904762,	32.34285714285715,	35.39047619047619,	38.43809523809524,	43.161904761904765,	48.49523809523809,	51.542857142857144,	56.266666666666666,	65.10476190476192,	72.87619047619047,	78.20952380952382,	83.6952380952381,	97.86666666666667])
taus_data = np.array([0.46367713,0.443497758,0.423318386,0.401793722,0.404484305	,0.452914798,0.471748879,	0.455605381,	0.347982063,	0.306278027,	0.225560538,	0.186547085	,0.187892377	,0.165022422	,0.109865471]) # s
SDs = np.array([0.49478673,	0.411374408,	0.4	,0.35450237	,0.381042654,	0.407582938	,0.636966825,	0.472037915,	0.411374408	,0.301421801,	0.263507109	,0.225592417,	0.193364929	,0.172511848,	0.106161137])
invSlopes = np.array([0.508056872	,0.437914692,	0.426540284,	0.436018957,	0.420853081	,0.447393365,	0.678672986,	0.51563981,	0.447393365	,0.312796209	,0.276777251	,0.244549763	,0.206635071	,0.172511848	,0.111848341])

file = np.load('E_sLex_aug.npz')
fs = file['fs']
taus = file['taus']

plt.figure(figsize=(3,2))

plt.plot(fs_data,taus_data,'D',color='k',markersize=5)
plt.plot(fs_data,SDs,'^',color='#d90429',markersize=4)
plt.plot(fs_data,invSlopes,'s',color='#118ab2',markersize=3)
plt.plot(fs,taus,color='k')


plt.xlim(0,110)
plt.ylim(0,.7)
plt.gca().set_xticks([0,50,100])
plt.gca().set_xticklabels([0,'',100])
plt.gca().set_yticks([0,0.2,0.4,.6])
plt.gca().set_yticklabels([0,'','',.6])


#Inset
tuned_D_file = np.load('D_aug.npz')
Es = tuned_D_file['Es']
Ds = tuned_D_file['Ds']
D_adjustments = tuned_D_file['D_adjustments']
plt.figure(figsize=(1.6,1.2))
plt.plot(Es,Ds/np.max(Ds),color='k')
plt.plot(Es,(Ds+D_adjustments)/np.max(Ds),color=alt_color)
plt.tick_params(pad=1.2,length=2.5)
plt.subplots_adjust(wspace=0.2,hspace=0.13,bottom=.2,left=0.2)
plt.xlim(min(Es),max(Es))
plt.ylim(0,1)
plt.gca().set_xticks([4,5.5])
plt.gca().set_xticklabels([4,5.5])
plt.gca().set_yticks([0,.5,1])
plt.gca().set_yticklabels([0,'',1])



plt.show()


