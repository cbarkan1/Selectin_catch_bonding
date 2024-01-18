# P-selectin -- sPSGL-1

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi

# Experimental data:
fs_sP = np.array([4.75297619047619,	6.74867724867725,	8.73875661375661,	10.92691798941799,	14.070767195767196,	17.611441798941797,	26.80820105820106,	36.088955026455025])
taus_sP = np.array([0.106666667,	0.265,	0.395,	0.623333333,	0.268333333,	0.113333333,	0.115,	0.09])
SDs_sP = np.array([0.103892613,	0.335503363,	0.405707018	,0.649527099,	0.44570724,	0.299775074,	0.233476805	,0.112356505])
invSlopes = np.array([0.125363825,	0.391060291	,0.454677755,	0.703534304	,0.361122661,	0.052390852,	0.084199584, np.nan])



W = 2.8
a = 1.7 #1/nm
sigma = 0.31 # radians
D0 = 166.7 # pNnm
k_theta = 240 # 250 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.58*pi
theta1 = 0.9*pi

xmin,xmax=4.3,2*W*0.99999
ymin,ymax = -.2, 2.1

left_transmissive = True
top_transmissive = True
right_transmissive = False
bottom_transmissive = False

Pselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)


fs = np.linspace(0,50,100)
num_fs = len(fs)
taus = np.zeros(num_fs)
Ebs = np.zeros(num_fs)
taus_langer = np.zeros(num_fs)



for i in range(num_fs):
	time0 = time()
	Pselectin.f = fs[i]
	rate = escape_rate(Pselectin,xmin,xmax,65,ymin,ymax,65,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	taus_langer[i],Ebs[i] = Pselectin.langers_tau(xmin,xmax,ymin,ymax)
	time1 = time()
	print('iteration time = ',time1-time0)


plt.plot(fs,taus)
plt.plot(fs_sP,taus_sP,'s')
plt.plot(fs_sP,SDs_sP,'o')
plt.plot(fs_sP,invSlopes,'^')
plt.show()



