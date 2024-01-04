import numpy as np
from numpy.linalg import eigh
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from bonds import *
pi = np.pi

ell = np.array([1,1])
def plot_detH0_curve(ax,bond,xmin,xmax,ymin,ymax,detH_width=3):
	x_grid = np.linspace(xmin,xmax,300)
	y_grid = np.linspace(ymax,ymin,300)
	x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
	detHs = bond.detH(x_mesh,y_mesh)
	boundary = ax.contour(x_mesh, y_mesh,detHs, [0],colors=['#49be25'],linewidths=detH_width)

	boundary_points_collection = []
	for p in boundary.collections[0].get_paths():
		boundary_points_collection += [p.vertices]

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

	return switch_points,boundary_points_collection

def adjugate(M):
	# Adjugate of 2x2 matrix
	return np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]])

def plot_vector_field():
	x_vector_range = np.linspace(xmin*1.004,xmax*0.997,16)
	y_vector_range = np.linspace(ymin*.96,ymax*0.97,16)
	x_vector_mesh,y_vector_mesh = np.meshgrid(x_vector_range,y_vector_range)
	vector_field = np.zeros((len(x_vector_range),len(y_vector_range),2))
	for i in range(len(x_vector_range)):
		for ii in range(len(y_vector_range)):
			x,y = x_vector_range[i],y_vector_range[ii]
			H = selectin.hessian(x,y)
			vector = adjugate(H)@ell * np.sign(np.linalg.det(H))
			#vector[1,0]*=aspect_factor
			vector_field[i,ii,:] = vector[:]/np.linalg.norm(vector[:])
	plt.quiver(x_vector_mesh.T,y_vector_mesh.T,vector_field[:,:,0],vector_field[:,:,1],units='x',scale_units='x',scale=32,width=.004,headwidth=4,pivot='middle')
	return



def regularized_trajectory(bond,x0,y0,us):
	def dXdu(X,u):
		H = bond.hessian(X[0],X[1])
		return (adjugate(H)@ell).reshape(2,)

	Xs = odeint(dXdu,[x0,y0],us)
	return Xs


W = 2.8
a = 3.5 #1/nm
sigma = 0.4 # radians
D0 = 220 # pNnm
k_theta = 350 # pNnm
gamma = 0.000033 # pN s / nm
theta0 = 0.6*pi
theta1 = 1.*pi

xmin,xmax,ymin,ymax = 4.7,5.5999,-0.2,0.7
selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
selectin.find_minimum()
selectin.find_saddle(xmin,xmax,ymin,ymax)


plt.figure(figsize=(4.5,4.5))
ax = plt.gca()

switch_points,boundary_points_collection = plot_detH0_curve(ax,selectin,xmin,xmax,ymin,ymax,detH_width=5)

grey_color = '#cFcFcF'
left_minimumlike_points = np.concatenate((boundary_points_collection[0],np.array([[4,-1]])))
plt.rcParams["hatch.linewidth"] = 8
left_minimumlike_patch = Polygon(left_minimumlike_points,hatch='/',facecolor=grey_color,edgecolor='#FFFFFF',rasterized=True)
ax = plt.gca()
ax.add_patch(left_minimumlike_patch)

right_minimumlike_points = np.concatenate((boundary_points_collection[1],np.array([[6,-1]])))
right_minimumlike_patch = Polygon(right_minimumlike_points,facecolor=grey_color)
ax = plt.gca()
ax.add_patch(right_minimumlike_patch)


### Separatrix:
###
Sepx_linewidth = 3.3
Sepx_linestyle = (0,(2.5,.8))

us = np.linspace(0,-.007,2000)
X = regularized_trajectory(selectin,switch_points[0][0]+.01,switch_points[0][1]+.01,us)
ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

us = np.linspace(0,.008,100)
X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]+.01,us)
ax.plot(X[:,0],X[:,1],'--',color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

us = np.linspace(0,-.005,100)
X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]-.01,us)
ax.plot(X[:,0],X[:,1],'--',color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

###
###


###
### Finding switch line
###
SL_width = 3.1

selectin.h_force = [80,30]
fs = np.linspace(0,50,50)
num_fs = len(fs)
xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
Ebs = np.zeros(num_fs)
for i in range(num_fs):
	selectin.f = fs[i]
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)
	xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
	Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

catch_to_slip_index = np.argmax(Ebs)

H = selectin.hessian(switch_points[1][0],switch_points[1][1])
lambdas,vs = np.linalg.eigh(H)
v0 = vs[:,0]
vscale = 0.01
M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2]])
Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index]]
A = np.linalg.solve(M.T,Y)

switchline_xs = np.linspace(xmin,switch_points[1][0],100)
switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
selectin.h_force = [0,0]
selectin.f = 0
selectin.find_minimum()
selectin.find_saddle(xmin,xmax,ymin,ymax)
###
###
###


for point in switch_points:
	plt.plot(point[0],point[1],'o',color='#ff7700',markersize=18)


plt.plot([selectin.xm],[selectin.ym],'o',color='k',markersize=14)
plt.plot([selectin.xs],[selectin.ys],'X',color='k',markersize=14)


ax.set_xlim(1.7*W,xmax)
ax.set_ylim(-.1,.7)

ax.set_xticks([1.7*W,1.8*W,1.9*W,xmax])
ax.set_xticklabels(['']*4)
ax.set_yticks([0.01,1/a])
ax.set_yticklabels(['',''])

plot_vector_field()


ax.set_aspect(1.05, adjustable='box')


plt.show()
