# L-selectin -- PSGL-1

import numpy as np
from numpy import exp,sqrt,cos,log,dot,sign,arctan2,arccos,arcsin
from numpy.linalg import det,inv,eigh,eigvalsh,norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi

fs_data = [7.488210756540372,	14.211122712045562,	24.905018036429137,	34.597186416020506,	44.99893612283577,	54.68754175099336,	64.40464939062106,	74.3007852897952,	84.06242732234489] #pN
taus_data = [0.00654804,	0.015326264,	0.021569986,	0.060916617,	0.110128111,	0.152437763	,0.171043243,	0.140756887,	0.122324596] # s


W = 2.8
a = 3.5 #1/nm
sigma = 0.37 # radians
D0 = 237.3 # pNnm
k_theta = 266 # pNnm
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

fs = np.linspace(0,100,100)
num_fs = len(fs)
taus = np.zeros(num_fs)
for i in range(num_fs):
	time0 = time()
	Lselectin.f = fs[i]
	rate = escape_rate(Lselectin,xmin,xmax,90,ymin,ymax,90,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	time1 = time()
	print('iteration time = ',time1-time0)


plt.plot(fs,taus)
plt.plot(fs_data,taus_data,'s',color='k',markersize=5)
plt.show()


