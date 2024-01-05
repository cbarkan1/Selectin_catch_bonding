import numpy as np
from numpy.linalg import eigh
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from bonds import *
pi = np.pi

ell = np.array([1,1])
def plot_detH0_curve(bond,xmin,xmax,ymin,ymax):
	x_grid = np.linspace(xmin,xmax,300)
	y_grid = np.linspace(ymax,ymin,300)
	x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
	detHs = bond.detH(x_mesh,y_mesh)
	boundary = plt.contour(x_mesh, y_mesh,detHs, [0],colors=['#49be25'],linewidths=3)

	switch_points = []
	for curve in boundary.collections[0].get_paths():
		points = curve.vertices
		num_pts = points.shape[0]

		x = points[0,0]
		y = points[0,1]
		H = bond.hessian(x,y)
		evals,evects = eigh(H)
		zero_v_previous = evects[:,0]
		sigma_previous = np.dot(zero_v_previous,ell)

		for i in range(1,num_pts):
			x = points[i,0]
			y = points[i,1]
			H = bond.hessian(x,y)
			evals,evects = eigh(H)
			zero_v = evects[:,0]*np.sign(np.dot(evects[:,0],zero_v_previous))
			sigma = np.dot(zero_v,ell)
			if sigma*sigma_previous<0:
				switch_points += [[x,y]]
			sigma_previous = sigma
			zero_v_previous = zero_v

	return switch_points


def adjugate(M): # Adjugate of 2x2 matrix
	return np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])


def regularized_trajectory(bond,E0,d0,us):
	def dXdu(X,u):
		H = bond.hessian(X[0],X[1])
		return adjugate(H)@ell #(adjugate(H)@ell_vect).reshape(2,)
	Xs = odeint(dXdu,[E0,d0],us)
	return Xs


def plot_curve_with_arrow(ax,xs,ys,length_scale=5,angle=0.5,dv_adjust_angle=0,xy_scale_ratio=None,linewidth=1,color='k'):
	# xy_scale_ratio = x_scale = ((xmax-xmin)/width)  /  ((ymax-ymin)/height)

	def R(theta): # Rotation matrix
		return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])


	if xy_scale_ratio==None: # Calculate xy_scale_ratio from ax
		xmin,xmax = ax.get_xlim()
		ymin,ymax = ax.get_ylim()
		fig = ax.get_figure()
		bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		width, height = bbox.width, bbox.height
		xy_scale_ratio = ((xmax-xmin)/width)  /  ((ymax-ymin)/height)


	dx = xs[-1] - xs[-2]
	dy = ys[-1] - ys[-2]
	dv = R(dv_adjust_angle)@[dx,dy]
	T = np.array([[1/xy_scale_ratio,0],[0,1]])
	T_inv = np.array([[xy_scale_ratio,0],[0,1]])

	dv_prime = T@dv
	s1_prime = R(angle)@dv_prime
	s2_prime = R(-angle)@dv_prime

	s1 = length_scale*T_inv@s1_prime
	s2 = length_scale*T_inv@s2_prime

	xs = np.append(xs,[xs[-1] - s1[0] , xs[-1] , xs[-1] - s2[0] , xs[-1] - 0.5*s2[0]])
	ys = np.append(ys,[ys[-1] - s1[1] , ys[-1] , ys[-1] - s2[1] , ys[-1] - 0.5*s2[1]])

	plt.plot(xs,ys,linewidth=linewidth,color=color)




Sepx_linestyle = (0,(1,1))
Sepx_linewidth = 1.7
SL_width = 2


W = 2.8
a = 3.9 #1/nm
sigma = 0.57 # radians
D0 = 206.9 # pNnm
k_theta = 273 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.58*pi
theta1 = 0.995*pi

xmin,xmax=3.7,2*W*0.99999
ymin,ymax = -.4,1.1

Eselectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
def D_adjustment(self,theta):
	return 9.5
Eselectin.modify_D_adjustment(D_adjustment)

# Plots detH0 curve
plt.figure(figsize=(3,2))
plt.xlim(3.9,xmax)
plt.ylim(-.2,1.9)
ax = plt.gca()
ax.set_xticks([4.,4.5,5,5.5])
ax.set_xticklabels([4.,'','',5.5])
ax.set_yticks([0,.5,1,1.5])
ax.set_yticklabels([0,'','',1.5])



switch_points = plot_detH0_curve(Eselectin,3.7,xmax,ymin,ymax)


# Plot separatrix
E0,d0 = switch_points[0][:]
us = np.linspace(0,0.1,10000)
Xs = regularized_trajectory(Eselectin,E0,d0,us)
plt.plot(Xs[:,0],Xs[:,1],color='#5c5c5c',linestyle=Sepx_linestyle,linewidth=Sepx_linewidth)
Xs = regularized_trajectory(Eselectin,E0-.01,d0-.01,-us[0:1200])
plt.plot(Xs[:,0],Xs[:,1],color='#5c5c5c',linestyle=Sepx_linestyle,linewidth=Sepx_linewidth)
#Xs = regularized_trajectory(Lselectin,E0-.01,d0+.01,us)
#plt.plot(Xs[:,0],Xs[:,1],'--',color='#5c5c5c')


# Plot switch points:
for point in switch_points:
	plt.plot([point[0]],[point[1]],'o',color='#ff7700',markersize=14)


fs1 = np.linspace(0.1,4,100)
fs2 = np.linspace(4,230,100)
fs = np.concatenate((fs1,fs2))
#fs = fs2
num_fs = len(fs)
Ebs = np.zeros(num_fs)
saddles,minimums = np.zeros((num_fs,2)),np.zeros((num_fs,2))
for i in range(num_fs):
	Eselectin.f = fs[i]
	Eselectin.find_saddle_precision(xmin,xmax,ymin,ymax)
	#Eselectin.find_saddle(xmin,xmax,ymin,ymax)
	Eselectin.find_minimum()
	saddles[i,0],saddles[i,1] = Eselectin.xs,Eselectin.ys
	minimums[i,0],minimums[i,1] = Eselectin.xm,Eselectin.ym
	Ebs[i] = Eselectin.V(Eselectin.xs,Eselectin.ys) - Eselectin.V(Eselectin.xm,Eselectin.ym)



if 1: # Finding switch line

	def find_switch(bond,h_force,fs,switch_type):
		bond.h_force = h_force
		num_fs = len(fs)
		xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
		Ebs = np.zeros(num_fs)
		for i in range(num_fs):
			bond.f = fs[i]
			bond.find_minimum()
			bond.find_saddle(xmin,xmax,ymin,ymax)
			xms[i],yms[i],xss[i],yss[i] = bond.xm,bond.ym,bond.xs,bond.ys
			Ebs[i] = bond.V(xss[i],yss[i]) - bond.V(xms[i],yms[i])

		

		if switch_type=='catch_to_slip':
			switch_index = np.argmax(Ebs)
		elif switch_type=='slip_to_catch':
			switch_index = np.argmin(Ebs)

		x_switch,y_switch = xss[switch_index],yss[switch_index]
		bond.h_force = [0,0]
		return x_switch,y_switch


	slip_to_catch_index = np.argmin(Ebs[0:110])
	xs_slip_to_catch, ys_slip_to_catch = saddles[slip_to_catch_index,0], saddles[slip_to_catch_index,1]
	catch_to_slip_index = np.argmax(Ebs[110:])+110
	xs_catch_to_slip, ys_catch_to_slip = saddles[catch_to_slip_index,0], saddles[catch_to_slip_index,1]
	#plt.plot(xs_catch_to_slip,ys_catch_to_slip)

	x1,y1 = find_switch(Eselectin,[5,100],np.linspace(0,100,50),'catch_to_slip')
	x2,y2 = find_switch(Eselectin,[-20,-3],np.linspace(0,10,50),'slip_to_catch')
	x3,y3 = find_switch(Eselectin,[-35,-3],np.linspace(2,20,50),'slip_to_catch')

	x_switches = np.array([xs_slip_to_catch,xs_catch_to_slip,x1,x2,x3])
	y_switches = np.array([ys_slip_to_catch,ys_catch_to_slip,y1,y2,y3])


	H = Eselectin.hessian(switch_points[1][0],switch_points[1][1])
	lambdas,vs = np.linalg.eigh(H)
	v0 = vs[:,0]
	vscale = 0.5
	x_switches = np.append(x_switches,[switch_points[1][0],switch_points[1][0]+vscale*v0[0]])
	y_switches = np.append(y_switches,[switch_points[1][1],switch_points[1][1]+vscale*v0[1]])

	reordered_indices = np.argsort(x_switches)
	x_switches = x_switches[reordered_indices]
	y_switches = y_switches[reordered_indices]

	from scipy.interpolate import CubicSpline
	switchline_xs = np.linspace(xmin,switch_points[1][0],100)
	switchline_func = CubicSpline(x_switches,y_switches)
	plt.plot(switchline_xs,switchline_func(switchline_xs),color='#ff7700',linewidth=SL_width)


plt.plot([minimums[0,0]],[minimums[0,1]],'ko',markersize=9)
plt.plot([saddles[0,0]],[saddles[0,1]],'kX',markersize=10)


plot_curve_with_arrow(ax,minimums[:,0],minimums[:,1],length_scale=120,linewidth=1.7,angle=0.45,dv_adjust_angle=-.36)
plot_curve_with_arrow(ax,saddles[:,0],saddles[:,1],length_scale=41,linewidth=1.7,angle=0.45)



plt.show()





