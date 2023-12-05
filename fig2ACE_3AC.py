import numpy as np
import matplotlib.pyplot as plt


alt_color = '#e07a5f'

if 1: # L-selectin--PSGL1  and LSelN138G--PSGL1
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
	plt.savefig('L-PSGL1_plot.eps',transparent=True)

	plt.show()




if 0: # L-selectin--2GSP6  and  LSelA108H--2GSP6
	fs_data_WT = np.array([12.000000000000007,	16.846153846153847,	22.15384615384616,	28.846153846153847,	36.92307692307693,	43.38461538461538,	55.15384615384617,	78,	103.84615384615384])
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
	plt.savefig('L-2GSP6_plot.eps',transparent=True)

	plt.show()



if 0: # P-selectin--PSGL1
	fs_data = np.array([4.75297619047619,	6.74867724867725,	8.73875661375661,	10.92691798941799,	14.070767195767196,	17.611441798941797,	26.80820105820106,	36.088955026455025])
	taus_data = np.array([0.106666667,	0.265,	0.395,	0.623333333,	0.268333333,	0.113333333,	0.115,	0.09])
	SDs = np.array([0.103892613,	0.335503363,	0.405707018	,0.649527099,	0.44570724,	0.299775074,	0.233476805	,0.112356505])
	invSlopes = np.array([0.125363825,	0.391060291	,0.454677755,	0.703534304	,0.361122661,	0.052390852,	0.084199584, np.nan])

	file = np.load('P-PSGL1.npz')
	fs = file['fs']
	taus = file['taus']

	plt.figure(figsize=(3,2))

	plt.plot(fs,taus,color='k')
	plt.plot(fs_data,taus_data,'D',color='k',markersize=5)
	plt.plot(fs_data,SDs,'^',color='#d90429',markersize=4)
	plt.plot(fs_data,invSlopes,'s',color='#118ab2',markersize=3)
	
	

	plt.xlim(0,40)
	plt.ylim(0,.75)
	plt.gca().set_xticks([0,10,20,30,40])
	plt.gca().set_xticklabels([0,'','','',40])
	plt.gca().set_yticks([0,0.2,0.4,.6])
	plt.gca().set_yticklabels([0,'','',.6])
	plt.savefig('P_plot.eps',transparent=True)

	plt.show()



if 0: # E-selectin--sLex Untuned
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
	plt.savefig('E_untuned_plot.eps',transparent=True)

	plt.show()


if 0: # E-selectin--sLex Tuned
	fs_data= np.array([12.838095238095235,	23.352380952380948,	27.61904761904762,	32.34285714285715,	35.39047619047619,	38.43809523809524,	43.161904761904765,	48.49523809523809,	51.542857142857144,	56.266666666666666,	65.10476190476192,	72.87619047619047,	78.20952380952382,	83.6952380952381,	97.86666666666667])
	taus_data = np.array([0.46367713,0.443497758,0.423318386,0.401793722,0.404484305	,0.452914798,0.471748879,	0.455605381,	0.347982063,	0.306278027,	0.225560538,	0.186547085	,0.187892377	,0.165022422	,0.109865471]) # s
	SDs = np.array([0.49478673,	0.411374408,	0.4	,0.35450237	,0.381042654,	0.407582938	,0.636966825,	0.472037915,	0.411374408	,0.301421801,	0.263507109	,0.225592417,	0.193364929	,0.172511848,	0.106161137])
	invSlopes = np.array([0.508056872	,0.437914692,	0.426540284,	0.436018957,	0.420853081	,0.447393365,	0.678672986,	0.51563981,	0.447393365	,0.312796209	,0.276777251	,0.244549763	,0.206635071	,0.172511848	,0.111848341])

	file = np.load('E_sLex_tuned.npz')
	fs = file['fs']
	taus = file['taus']

	plt.figure(figsize=(3,2))


	#plt.plot(fs[50],taus[50],'o')
	#plt.plot(fs[55],taus[55],'o')
	#plt.plot(fs[60],taus[60],'o')

	delete_indices = [38,39,40,55,56]
	fs = np.delete(fs,delete_indices)
	taus = np.delete(taus,delete_indices)



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
	plt.savefig('E_tuned_plot.eps',transparent=True)

	"""
	tuned_D_file = np.load('../Geometric_Framework_Code/Numerical Lifetimes/E_tuned_D.npz')
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
	#plt.savefig('D_plot.eps',transparent=True)
	"""
	plt.show()

#


