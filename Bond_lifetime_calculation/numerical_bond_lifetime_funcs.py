#Functions for solving steady state Fokker-Planck equation to compute mean bond lifetime

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from time import time

def M_matrix(bond,x_min,x_max,num_x,y_min,y_max,num_y,index_regen,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive):
	"""
	Index labeling:

	     |    |
	     | 01 |
	     |    |
	----------------
	|    |    |    |
	| 10 | 11 | 12 |
	|    |    |    |
	----------------
	     |    |    
	     | 21 |
	     |    |

	"""
	x_grid = np.linspace(x_min,x_max,num_x)
	y_grid = np.linspace(y_max,y_min,num_y)
	ax = x_grid[1] - x_grid[0]
	ay = y_grid[0] - y_grid[1]

	T = bond.T

	T_matrix = np.zeros((num_x*num_y,num_x*num_y))
	exit_rates = np.zeros(num_x*num_y) #Diagonal of open_T_matrix

	#Interior
	for i in range(1,num_y-1):
		for ii in range(1,num_x-1):
			x11,y11 = x_grid[ii], y_grid[i]
			x01,y01 = x_grid[ii],y_grid[i-1]
			x10,y10 = x_grid[ii-1],y_grid[i]
			x12,y12 = x_grid[ii+1],y_grid[i]
			x21,y21 = x_grid[ii],y_grid[i+1]

			index11 = ii*num_y + i
			index01 = ii*num_y + i-1
			index10 = (ii-1)*num_y + i
			index12 = (ii+1)*num_y + i
			index21 = ii*num_y + i+1

			F11x,F11y = bond.force(x11,y11)
			F01x,F01y = bond.force(x01,y01)
			F10x,F10y = bond.force(x10,y10)
			F12x,F12y = bond.force(x12,y12)
			F21x,F21y = bond.force(x21,y21)

			rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
			rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
			rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
			rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

			T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
			T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
			T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
			T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
			T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

	#left boundary
	ii = 0
	for i in range(1,num_y-1):
		x11,y11 = x_grid[ii], y_grid[i]
		x01,y01 = x_grid[ii],y_grid[i-1]
		x12,y12 = x_grid[ii+1],y_grid[i]
		x21,y21 = x_grid[ii],y_grid[i+1]

		index11 = ii*num_y + i
		index01 = ii*num_y + i-1
		index12 = (ii+1)*num_y + i
		index21 = ii*num_y + i+1

		F11x,F11y = bond.force(x11,y11)
		F01x,F01y = bond.force(x01,y01)
		F10x,F10y = F11x,F11y
		F12x,F12y = bond.force(x12,y12)
		F21x,F21y = bond.force(x21,y21)

		rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
		rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2) if left_transmissive else 0
		rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
		rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

		T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
		T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
		T_matrix[index_regen,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
		T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
		T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

		exit_rates[index11] = -rate_11to10


	#right boundary
	ii = num_x-1
	for i in range(1,num_y-1):
		x11,y11 = x_grid[ii], y_grid[i]
		x01,y01 = x_grid[ii],y_grid[i-1]
		x10,y10 = x_grid[ii-1],y_grid[i]
		x21,y21 = x_grid[ii],y_grid[i+1]

		index11 = ii*num_y + i
		index01 = ii*num_y + i-1
		index10 = (ii-1)*num_y + i
		index21 = ii*num_y + i+1

		F11x,F11y = bond.force(x11,y11)
		F01x,F01y = bond.force(x01,y01)
		F10x,F10y = bond.force(x10,y10)
		F12x,F12y = F11x,F11y
		F21x,F21y = bond.force(x21,y21)

		rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
		rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
		rate_11to12 = max(0,F12x/(2*ax) + T/ax**2) if right_transmissive else 0
		rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

		T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
		T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
		T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
		T_matrix[index_regen,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
		T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

		exit_rates[index11] = -rate_11to12

	#top boundary
	i = 0
	for ii in range(1,num_x-1):
		x11,y11 = x_grid[ii], y_grid[i]
		x10,y10 = x_grid[ii-1],y_grid[i]
		x12,y12 = x_grid[ii+1],y_grid[i]
		x21,y21 = x_grid[ii],y_grid[i+1]

		index11 = ii*num_y + i
		index10 = (ii-1)*num_y + i
		index12 = (ii+1)*num_y + i
		index21 = ii*num_y + i+1

		F11x,F11y = bond.force(x11,y11)
		F01x,F01y = F11x,F11y
		F10x,F10y = bond.force(x10,y10)
		F12x,F12y = bond.force(x12,y12)
		F21x,F21y = bond.force(x21,y21)

		rate_11to01 = max(0,F01y/(2*ay) + T/ay**2) if top_transmissive else 0
		rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
		rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
		rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

		T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
		T_matrix[index_regen,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
		T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
		T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
		T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

		exit_rates[index11] = -rate_11to01

	#bottom boundary
	for ii in range(1,num_x-1):
		# Reflective
		i = num_y-1
		x11,y11 = x_grid[ii], y_grid[i]
		x01,y01 = x_grid[ii],y_grid[i-1]
		x10,y10 = x_grid[ii-1],y_grid[i]
		x12,y12 = x_grid[ii+1],y_grid[i]

		index11 = ii*num_y + i
		index01 = ii*num_y + i-1
		index10 = (ii-1)*num_y + i
		index12 = (ii+1)*num_y + i

		F11x,F11y = bond.force(x11,y11)
		F01x,F01y = bond.force(x01,y01)
		F10x,F10y = bond.force(x10,y10)
		F12x,F12y = bond.force(x12,y12)
		F21x,F21y = F11x,F11y

		rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
		rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
		rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
		rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2) if bottom_transmissive else 0

		T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
		T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
		T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
		T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
		T_matrix[index_regen,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

		exit_rates[index11] = -rate_11to21

	#Corners
	i,ii = 0,0
	x11,y11 = x_grid[ii], y_grid[i]
	x12,y12 = x_grid[ii+1],y_grid[i]
	x21,y21 = x_grid[ii],y_grid[i+1]

	index11 = ii*num_y + i
	index12 = (ii+1)*num_y + i
	index21 = ii*num_y + i+1

	F11x,F11y = bond.force(x11,y11)
	F01x,F01y = F11x,F11y
	F10x,F10y = F11x,F11y
	F12x,F12y = bond.force(x12,y12)
	F21x,F21y = bond.force(x21,y21)

	rate_11to01 = max(0,F01y/(2*ay) + T/ay**2) if top_transmissive else 0
	rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2) if left_transmissive else 0
	rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
	rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

	T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
	T_matrix[index_regen,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
	T_matrix[index_regen,index11] += rate_11to10 # -F10x/ax + (T/gamma)/ax**2
	T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
	T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

	exit_rates[index11] = -rate_11to01-rate_11to10


	i,ii = 0,num_x-1
	x11,y11 = x_grid[ii], y_grid[i]
	x10,y10 = x_grid[ii-1],y_grid[i]
	x21,y21 = x_grid[ii],y_grid[i+1]

	index11 = ii*num_y + i
	index10 = (ii-1)*num_y + i
	index21 = ii*num_y + i+1

	F11x,F11y = bond.force(x11,y11)
	F01x,F01y = F11x,F11y
	F10x,F10y = bond.force(x10,y10)
	F12x,F12y = F11x,F11y
	F21x,F21y = bond.force(x21,y21)

	rate_11to01 = max(0,F01y/(2*ay) + T/ay**2) if top_transmissive else 0
	rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
	rate_11to12 = max(0,F12x/(2*ax) + T/ax**2) if right_transmissive else 0
	rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2)

	T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
	T_matrix[index_regen,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
	T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
	T_matrix[index_regen,index11] += rate_11to12 # F12x/ax + (T/gamma)/ax**2
	T_matrix[index21,index11] = rate_11to21 # -F21y/ay + (T/gamma)/ay**2

	exit_rates[index11] = -rate_11to01-rate_11to12


	i,ii = num_y-1,0
	x11,y11 = x_grid[ii], y_grid[i]
	x01,y01 = x_grid[ii],y_grid[i-1]
	x12,y12 = x_grid[ii+1],y_grid[i]

	index11 = ii*num_y + i
	index01 = ii*num_y + i-1
	index12 = (ii+1)*num_y + i

	F11x,F11y = bond.force(x11,y11)
	F01x,F01y = bond.force(x01,y01)
	F10x,F10y = F11x,F11y
	F12x,F12y = bond.force(x12,y12)
	F21x,F21y = F11x,F11y

	rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
	rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2) if left_transmissive else 0
	rate_11to12 = max(0,F12x/(2*ax) + T/ax**2)
	rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2) if bottom_transmissive else 0

	T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
	T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
	T_matrix[index_regen,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
	T_matrix[index12,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
	T_matrix[index_regen,index11] += rate_11to21 # -F21y/ay + (T/gamma)/ay**2

	exit_rates[index11] = -rate_11to10-rate_11to21



	#Reflective
	i,ii = num_y-1,num_x-1
	x11,y11 = x_grid[ii], y_grid[i]
	x01,y01 = x_grid[ii],y_grid[i-1]
	x10,y10 = x_grid[ii-1],y_grid[i]

	index11 = ii*num_y + i
	index01 = ii*num_y + i-1
	index10 = (ii-1)*num_y + i

	F11x,F11y = bond.force(x11,y11)
	F01x,F01y = bond.force(x01,y01)
	F10x,F10y = bond.force(x10,y10)
	F12x,F12y = F11x,F11y
	F21x,F21y = F11x,F11y

	rate_11to01 = max(0,F01y/(2*ay) + T/ay**2)
	rate_11to10 = max(0,-F10x/(2*ax) + T/ax**2)
	rate_11to12 = max(0,F12x/(2*ax) + T/ax**2) if right_transmissive else 0
	rate_11to21 = max(0,-F21y/(2*ay) + T/ay**2) if bottom_transmissive else 0

	T_matrix[index11,index11] = -1*(rate_11to01+rate_11to10+rate_11to12+rate_11to21)#(F10x - F12x)/ax + (F21y-F01y)/ay - (2*T/gamma)*(1/ax**2 + 1/ay**2)
	T_matrix[index01,index11] = rate_11to01 #F01y/ay + (T/gamma)/ay**2
	T_matrix[index10,index11] = rate_11to10 # -F10x/ax + (T/gamma)/ax**2
	T_matrix[index_regen,index11] = rate_11to12 # F12x/ax + (T/gamma)/ax**2
	T_matrix[index_regen,index11] += rate_11to21 # -F21y/ay + (T/gamma)/ay**2

	exit_rates[index11] = -rate_11to12-rate_11to21

	return T_matrix,exit_rates



def escape_rate(bond,xmin,xmax,num_x,ymin,ymax,num_y,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive,sparse_method='coo'):


	bond.find_minimum()
	regen_i = round((ymax - bond.ym)/(ymax-ymin) * num_y)
	regen_ii = round((bond.xm - xmin)/(xmax-xmin) * num_x)
	regen_index = regen_ii*num_y+regen_i
	if regen_index > num_x*num_y:
		regen_index -= num_y


	T_matrix,exit_rates = M_matrix(bond,xmin,xmax,num_x,ymin,ymax,num_y,regen_index,left_transmissive,top_transmissive,right_transmissive,bottom_transmissive)


	if sparse_method=='coo':
		sparseT = sparse.coo_array(T_matrix)
	elif sparse_method=='bsr':
		sparseT = sparse.bsr_array(T_matrix)
	else:
		print('Error: sparse_method not recognized.')
		quit()



	time0 = time()
	evals,evects = sparse.linalg.eigs(sparseT,k=1,which='SM')
	time1 = time()
	print('eigs time = ',time1-time0)


	stationary_index = np.argmin(np.abs(evals))

	#Stationary distribution
	P = np.real(evects[:,stationary_index])

	rate = ( -1*np.sum(exit_rates*P)/np.sum(P) )/bond.gamma
	
	return rate