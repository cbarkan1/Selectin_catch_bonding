# LSelN138G -- PSGL-1

import numpy as np
from numpy import exp,sqrt,cos,log,dot,sign,arctan2,arccos,arcsin
from numpy.linalg import det,inv,eigh,eigvalsh,norm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi


fs_data = [7.179487179487186,	13.846153846153854,	26.66666666666667,	35.72649572649573,	44.44444444444444,	54.871794871794876,	64.27350427350426,	74.70085470085469]
taus_data = [0.003508772,	0.012631579	,0.092631579	,0.130526316	,0.224561404	,0.190877193,	0.169122807	,0.133333333]



W = 2.8
a = 3.5 #1/nm
sigma = 0.37 # radians
D0 = 210 # pNnm
k_theta = 220 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.58*pi
theta1 = 1.*pi
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
	rate = escape_rate(Lselectin,xmin,xmax,50,ymin,ymax,50,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	time1 = time()
	print('iteration time = ',time1-time0)


plt.plot(fs,taus)
plt.plot(fs_data,taus_data,'s',color='k',markersize=5)
plt.show()



