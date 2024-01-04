# LSelA108H -- 2-GSP-6

import numpy as np
from numpy import exp,sqrt,cos,log,dot,sign,arctan2,arccos,arcsin
from numpy.linalg import det,inv,eigh,eigvalsh,norm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi


fs_data = np.array([7.831094049904031,	12.898272552783112,	19.577735124760082,	29.251439539347423,	36.852207293666034,	44.68330134357007,	54.12667946257198,	66.10364683301344,	78.0806142034549,	94.43378119001923])
taus_data = np.array([0.3284236,	0.298179365,	0.252218421,	0.242459225,	0.181996254	,0.131192801	,0.119020464	,0.114069005	,0.071677933,	0.02078871])



W = 2.8
a = 3. #1/nm
sigma = 0.54# radians
D0 = 201 # pNnm
k_theta = 266 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.58*pi
theta1 = .969*pi
Lselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)

xmin,xmax=3.9,5.599
ymin,ymax = -.2,1.4


left_transmissive = False
top_transmissive = True
right_transmissive = False
bottom_transmissive = False



fs = np.linspace(0,130,100)
num_fs = len(fs)
taus = np.zeros(num_fs)
for i in range(num_fs):
	time0 = time()
	Lselectin.f = fs[i]
	rate = escape_rate(Lselectin,xmin,xmax,80,ymin,ymax,80,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	time1 = time()
	print('iteration time = ',time1-time0)


plt.plot(fs,taus)
plt.plot(fs_data,taus_data,'s',color='k',markersize=5)
plt.show()



