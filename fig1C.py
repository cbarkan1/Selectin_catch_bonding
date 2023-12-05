import numpy as np
from numpy import exp,sqrt,cos,log,dot,sign,arctan2,arccos,arcsin
from numpy.linalg import det,inv,eigh,eigvalsh,norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
pi = np.pi



c0_vect = np.array([[0.0],[0]]) #h vector
def V(x,y,f,params):
	#x: Extension E (E=2Wsin(theta/2))
	#y: Bond distance d
	W,theta0,sigma,D0,k_theta,a,theta1 = params[:]
	theta = 2*arcsin(x/(2*W))
	V_theta = 0.5*k_theta*(theta-theta0)**2
	D = D0*exp(-0.5*(theta-theta1)**2/sigma**2)
	M = (1-exp(-a*y))**2 - 1 
	return V_theta + D*M - y*(f+c0_vect[1,0]) - x*(f+c0_vect[0,0]) #- 2*4.28*log(y+4)

def grad_V_numerical(x,y,f,params):
	h = 0.00001
	V0 = V(x,y,f,params)
	V_x = (V(x+h,y,f,params) - V0)/h
	#print('Numerical V_x = ',V_x)
	V_y = (V(x,y+h,f,params) - V0)/h
	#print('Numerical V_y = ',V_y)
	return np.array([[V_x],[V_y]])

def Hessian_numerical(x,y,params):
	h = 0.00001
	V0 = V(x,y,0,params)
	V_xx = (V(x+h,y,0,params)+V(x-h,y,0,params)-2*V0)/h**2
	#print('Numerical V_xx = ',V_xx)
	V_yy = (V(x,y+h,0,params)+V(x,y-h,0,params)-2*V0)/h**2
	#print('Numerical V_yy = ',V_yy)
	V_xy = (V(x+h,y+h,0,params)+V0-V(x+h,y,0,params)-V(x,y+h,0,params))/h**2
	#print('Numerical V_xy = ',V_xy)
	return np.array([[V_xx,V_xy],[V_xy,V_yy]])


def Hessian_determinant_numerical(x,y,params):
	h = 0.00001
	V0 = V(x,y,0,params)
	V_xx = (V(x+h,y,0,params)+V(x-h,y,0,params)-2*V0)/h**2
	#print('Numerical V_xx = ',V_xx)
	V_yy = (V(x,y+h,0,params)+V(x,y-h,0,params)-2*V0)/h**2
	#print('Numerical V_yy = ',V_yy)
	V_xy = (V(x+h,y+h,0,params)+V0-V(x+h,y,0,params)-V(x,y+h,0,params))/h**2
	#print('Numerical V_xy = ',V_xy)
	return V_xx*V_yy-V_xy**2

def find_critical_points(x_range,y_range,f,params):
	#x_range = np.linspace(0,10,300)
	#y_range = np.linspace(-4,4,300)
	num_x = len(x_range)
	num_y = len(y_range)

	x_mesh,y_mesh = np.meshgrid(x_range,y_range)
	#Es = E1(x_mesh,y_mesh)
	Vxs = V(x_mesh.T,y_mesh.T,f,params)
	#print(Vxs[100:106,82:88])
	#plt.imshow(Vxs[100:106,82:88].T)
	#plt.colorbar()
	#plt.show()
	#quit()

	neighboring_signs = np.zeros((num_x-2,num_y-2,8))
	neighboring_signs[:,:,0] = Vxs[1:-1,1:-1]>Vxs[1:-1,0:-2]
	neighboring_signs[:,:,1] = Vxs[1:-1,1:-1]>Vxs[0:-2,0:-2]
	neighboring_signs[:,:,2] = Vxs[1:-1,1:-1]>Vxs[0:-2,1:-1]
	neighboring_signs[:,:,3] = Vxs[1:-1,1:-1]>Vxs[0:-2,2:]
	neighboring_signs[:,:,4] = Vxs[1:-1,1:-1]>Vxs[1:-1,2:]
	neighboring_signs[:,:,5] = Vxs[1:-1,1:-1]>Vxs[2:,2:]
	neighboring_signs[:,:,6] = Vxs[1:-1,1:-1]>Vxs[2:,1:-1]
	neighboring_signs[:,:,7] = Vxs[1:-1,1:-1]>Vxs[2:,0:-2]
	sign_flips = np.zeros((num_x-2,num_y-2))
	for i in range(8):
		sign_flips += (neighboring_signs[:,:,i]!=neighboring_signs[:,:,i-1]).astype(int)


	### Dealing with clusters corresponding to a single minimum or saddle:
	candidate_minima = np.where(sign_flips==0)
	#print(candidate_minima)
	num_candidates = len(candidate_minima[0])
	minima = []
	minimum_points = []
	for i in range(num_candidates):
		candidate = [candidate_minima[0][i],candidate_minima[1][i]]
		for ii in range(i+1,num_candidates):
			future_candidate = [candidate_minima[0][ii],candidate_minima[1][ii]]
			if candidate[0]==future_candidate[0] or candidate[1]==future_candidate[1]:
				break
		else:
			minima += [candidate]
			#minimum_points += [[x_range[candidate[0]+1],y_range[candidate[1]+1]]]
			x = x_range[candidate[0]+1]
			y = y_range[candidate[1]+1]
			H = Hessian_numerical(x,y,params)
			correction = np.linalg.inv(H)@grad_V_numerical(x,y,f,params)
			minimum_points += [[x-correction[0,0],y-correction[1,0]]]

	candidate_saddles = np.where(sign_flips==4)
	#print(candidate_saddles)
	num_candidates = len(candidate_saddles[0])
	saddles = []
	saddle_points = []
	for i in range(num_candidates):
		candidate = [candidate_saddles[0][i],candidate_saddles[1][i]]
		x = x_range[candidate[0]+1]
		y = y_range[candidate[1]+1]
		H = Hessian_numerical(x,y,params)
		if det(H)>=0:
			continue
		for ii in range(i+1,num_candidates):
			future_candidate = [candidate_saddles[0][ii],candidate_saddles[1][ii]]
			if candidate[0]==future_candidate[0] or candidate[1]==future_candidate[1]:
				break
		else:
			saddles += [candidate]
			#saddle_points += [[x_range[candidate[0]+1],y_range[candidate[1]+1]]]
			correction = np.linalg.inv(H)@grad_V_numerical(x,y,f,params)
			saddle_points += [[x-correction[0,0],y-correction[1,0]]]
	###

	if 0: #plot
		print('Inside find_critical_points')
		print(minima)
		print(saddles)

		for minimum in minima:
			Vxs[minimum[0]+1,minimum[1]+1] = 10
		for saddle in saddles:
			Vxs[saddle[0]+1,saddle[1]+1] += 20

		plt.imshow(np.log(Vxs.T-np.min(Vxs.T)+0.1),origin='lower')
		plt.colorbar()
		plt.show()

	return minimum_points,saddle_points

def langevin_trajectory(x0,y0,f,params):
	timesteps = 500000
	dt = 1e-9
	ts = np.arange(0,timesteps*dt,dt)
	etas_x = (dt*2*T/gamma)**0.5 * np.random.normal(size=timesteps)
	etas_y = (dt*2*T/gamma)**0.5 * np.random.normal(size=timesteps)
	xs = np.zeros(timesteps)
	ys = np.zeros(timesteps)
	xs[0],ys[0] = x0,y0
	for i in range(1,timesteps):
		grad_V = grad_V_numerical(xs[i-1],ys[i-1],f,params)
		xs[i] = min(xs[i-1] - grad_V[0,0]*dt/gamma + etas_x[i], 2*W*0.9999)
		ys[i] = ys[i-1] - grad_V[1,0]*dt/gamma + etas_y[i]

	return xs,ys

x_range = np.linspace(4,5.6,400) # E
y_range = np.linspace(-.2,.6,400) # d
x_mesh,y_mesh = np.meshgrid(x_range,y_range)



W = 2.8
a = 3.5
sigma = 0.37 # radians
D0 = 240 # pNnm
k_theta = 250 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.56*pi
theta1 = 1.*pi

T = 4.28

params = [W,theta0,sigma,D0,k_theta,a,theta1]



if 1: # Main text figure (Fig. 1B)
	f = 0

	minimum_points,saddle_points = find_critical_points(x_range,y_range,f,params)
	V_saddle = V(saddle_points[0][0],saddle_points[0][1],f,params)
	V_min = V(minimum_points[0][0],minimum_points[0][1],f,params)
	print(minimum_points)
	print(saddle_points)

	print((V_saddle-V_min)/T)

	print('theta in bound state = ',2*arcsin(minimum_points[0][0]/(2*W))/pi)

	Vs = (V(x_mesh.T,y_mesh.T,f,params)-V_saddle)/T
	Vs = V(x_mesh.T,y_mesh.T,f,params)/T

	#xs,ys = langevin_trajectory(minimum_points[0][0],minimum_points[0][1],f,params)

	plt.figure(figsize=(2.7,3.7))
	ax = plt.gca()

	im = plt.contourf(x_mesh.T, y_mesh.T,np.clip(Vs,-100,20), 15,cmap='YlGnBu',zorder=-1)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="6%", pad="20%")

	plt.colorbar(im, cax=cax, orientation='horizontal', ticks=[-8, 0, 8,16])


	print(minimum_points)
	ax.plot(minimum_points[0][0]-.01,minimum_points[0][1],'o',color='black',markersize=10)
	ax.plot(saddle_points[0][0],saddle_points[0][1],'X',color='black',markersize=10)

	#ax.plot(xs,ys,color='red')

	ax.set_xticks([1.7*W,1.8*W,1.9*W,2*W])
	ax.set_xticklabels(['']*4)
	ax.set_yticks([0,1/a])
	ax.set_yticklabels(['',''])
	ax.set_xlim(1.7*W,2*W)
	ax.set_ylim(-.2,.6)
	ax.tick_params(pad=1.2,length=2.5)
	plt.subplots_adjust(wspace=0.2,hspace=0.13,bottom=.2)
	#ax.set_aspect(1., adjustable='box')
	plt.savefig('fig1_V.eps',transparent=True)
	plt.show()



if 0: #Supplemental figures
	fig, ax = plt.subplots(1,5,figsize=(14,3))

	fs = [0,30,60,90,250]
	
	levels = np.arange(-20,22,2)
	#print(levels)
	#quit()
	for i in range(5):
		f = fs[i]

		minimum_points,saddle_points = find_critical_points(x_range,y_range,f,params)
		V_saddle = V(saddle_points[0][0],saddle_points[0][1],f,params)
		Vs = (V(x_mesh.T,y_mesh.T,f,params) - V_saddle)/T
		im = ax[i].contourf(x_mesh.T, y_mesh.T,Vs, levels=levels,cmap='YlGnBu',zorder=-1)

		ax[i].plot(minimum_points[0][0]-.01,minimum_points[0][1],'o',color='black',markersize=10)
		ax[i].plot(saddle_points[0][0],saddle_points[0][1],'X',color='black',markersize=10)


		ax[i].set_xticks([1.7*W,1.8*W,1.9*W,2*W])
		ax[i].set_xticklabels(['']*4)
		ax[i].set_yticks([0,1/a])
		ax[i].set_yticklabels(['',''])
		ax[i].set_xlim(1.7*W,2*W)
		ax[i].set_ylim(-.2,.6)

		ax[i].tick_params(pad=2,length=3)

	#divider = make_axes_locatable(ax[-1])
	#cax = divider.append_axes("right", size="10%", pad="10%")
	#plt.colorbar(im, cax=cax, orientation='vertical', ticks=levels[::4])
	#plt.subplots_adjust(hspace=.32,wspace=0.15)

	plt.savefig('fig_SI_Vfs.eps',transparent=True)
	plt.show()


#

