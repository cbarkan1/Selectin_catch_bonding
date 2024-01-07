# E-selectin -- sLex

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numerical_bond_lifetime_funcs import *
from bonds import *
pi = np.pi


fs_data= [12.838095238095235,	23.352380952380948,	27.61904761904762,	32.34285714285715,	35.39047619047619,	38.43809523809524,	43.161904761904765,	48.49523809523809,	51.542857142857144,	56.266666666666666,	65.10476190476192,	72.87619047619047,	78.20952380952382,	83.6952380952381,	97.86666666666667] # pN
taus_data = np.array([0.46367713,0.443497758,0.423318386,0.401793722,0.404484305	,0.452914798,0.471748879,	0.455605381,	0.347982063,	0.306278027,	0.225560538,	0.186547085	,0.187892377	,0.165022422	,0.109865471]) # s
SDs = np.array([0.49478673,	0.411374408,	0.4	,0.35450237	,0.381042654,	0.407582938	,0.636966825,	0.472037915,	0.411374408	,0.301421801,	0.263507109	,0.225592417,	0.193364929	,0.172511848,	0.106161137])
invSlopes = np.array([0.508056872	,0.437914692,	0.426540284,	0.436018957,	0.420853081	,0.447393365,	0.678672986,	0.51563981,	0.447393365	,0.312796209	,0.276777251	,0.244549763	,0.206635071	,0.172511848	,0.111848341])


if 0: # Gaussian D(theta)
	W = 2.8
	a = 3.9 #1/nm
	sigma = .57  # 0.57 # radians
	D0 = 206.9 # 205 # pNnm
	k_theta = 273 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.58*pi
	theta1 = .995*pi

	xmin,xmax=3.7,2*W*0.99999
	ymin,ymax = -.2,1.5
	num_x,num_y = 51,51

	left_transmissive = False
	top_transmissive = True
	right_transmissive = False
	bottom_transmissive = False

	Eselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	def D_adjustment(self,theta):
		return 9.5 + 0*theta
	Eselectin.modify_D_adjustment(D_adjustment)

else: # Augmented D(theta)
	W = 2.8
	a = 3.9 #1/nm
	sigma = 0.57 # radians
	D0 = 206.9 # pNnm
	k_theta = 273 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.58*pi
	theta1 = .995*pi

	xmin,xmax=3.7,2*W*0.99999
	ymin,ymax = -.2,1.5
	num_x,num_y = 50,50

	left_transmissive = False
	top_transmissive = True
	right_transmissive = False
	bottom_transmissive = False

	x0 = [ 5.32681447,  8.499551  ,  5.20188129 , 0.03515406, 29.9928431 ,  4.65236583, 0.17830888] # it 21

	const,a1,mu1,s1,a2,mu2,s2 = x0[:]
	Eselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	def D_adjustment(self,theta):
		x = 2*W*np.sin(theta/2)
		return const + a1*np.exp(-0.5*(x-mu1)**2/s1**2) + a2*np.exp(-0.5*(x-mu2)**2/s2**2)
	Eselectin.modify_D_adjustment(D_adjustment)


fs = np.linspace(5,70,14)
num_fs = len(fs)
taus = np.zeros(num_fs)
for i in range(num_fs):
	time0 = time()
	f = fs[i]
	Eselectin.f = f
	rate = escape_rate(Eselectin,xmin,xmax,num_x,ymin,ymax,num_y,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)
	taus[i] = 1/rate
	time1 = time()
	print('iteration time = ',time1-time0)

plt.plot(fs,taus,'.-')
plt.plot(fs_data,taus_data,'s')
plt.plot(fs_data,SDs,'o')
plt.plot(fs_data,invSlopes,'^')
plt.show()


