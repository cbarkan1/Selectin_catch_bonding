# L-selectin -- 2-GSP-6

import numpy as np
from numpy import exp,sqrt,cos,log,dot,sign,arctan2,arccos,arcsin
from numpy.linalg import det,inv,eigh,eigvalsh,norm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi


fs_data = np.array([12.0,	16.846153846153847,	22.15384615384616,	28.846153846153847,	36.92307692307693,	43.38461538461538,	55.15384615384617,	78,	103.84615384615384])
taus_data = np.array([0.026634383,	0.037530266	,0.064164649,	0.095641646	,0.163438257,	0.129539952	,0.071428571,	0.049636804	,0.041162228])



W = 2.8
a = 3. #1/nm
sigma = 0.298# radians
D0 = 217 # pNnm
k_theta = 266 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.58*pi
theta1 = .979*pi
Lselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)

xmin,xmax=4.9,5.599
ymin,ymax = -.2,.65


left_transmissive = True
top_transmissive = True
right_transmissive = False
bottom_transmissive = False


fs = np.linspace(0,130,100)
num_fs = len(fs)
taus = np.zeros(num_fs)
for i in range(num_fs):
	time0 = time()
	Lselectin.f = fs[i]
	rate = escape_rate(Lselectin,xmin,xmax,60,ymin,ymax,60,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	time1 = time()
	print('iteration time = ',time1-time0)


plt.plot(fs,taus)
plt.plot(fs_data,taus_data,'s',color='k',markersize=5)
plt.show()



