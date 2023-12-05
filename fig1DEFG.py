"""
Updated version of figure 1 for PNAS revision.

- For the three scenarios (slip, catch-slip, slip-catch-slip), it's going to have deformed det(H) curve
- Using the "Selectin" bond Class, instead of params


"""

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


def plot_contours(bond,xmin,xmax,ymin,ymax):
	xs = np.linspace(xmin,xmax,400)
	ys = np.linspace(ymin,ymax,400)
	x_mesh,y_mesh = np.meshgrid(xs,ys)
	Vs = bond.V(x_mesh.T,y_mesh.T)

	plt.figure()
	#plt.contour(x_mesh.T, y_mesh.T,np.clip(detHs,-1,1), 30)
	c = plt.contour(x_mesh.T, y_mesh.T,Vs, 35,zorder=-1)
	#plt.contour(x_mesh.T, y_mesh.T,np.clip(fs,-1900,100), 70)
	plt.xlabel('x')
	plt.ylabel('y')
	#plt.colorbar()
	#plt.gca().set_aspect('equal', adjustable='box')
	return


def plot_arrow(ax,x_traj,y_traj,scale=20,linewidth=1,shift=0,angle=0.5):
	dx = x_traj[-1] - x_traj[-2]
	dy = y_traj[-1] - y_traj[-2]
	velocity = [dx,dy]
	def R(theta): # Rotation matrix
		return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

	side1 = -1*R(angle)@velocity / np.linalg.norm(velocity)
	side2 = -1*R(-angle)@velocity / np.linalg.norm(velocity)
	
	ax.plot([x_traj[-1]+scale*side2[0]+shift*dx,x_traj[-1]+shift*dx,x_traj[-1]+scale*side1[0]+shift*dx],[y_traj[-1]+scale*side2[1]+shift*dy,y_traj[-1]+shift*dy,y_traj[-1]+scale*side1[1]+shift*dy],'k',linewidth=linewidth)

def plot_arrow_special(ax,x_traj,y_traj,scale=20,linewidth=1,shift=0,angle=0.5):
	dx = x_traj[-1] - x_traj[-2]
	dy = y_traj[-1] - y_traj[-2]
	velocity = [dx,dy]
	def R(theta): # Rotation matrix
		return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

	side1 = -1.15*R(angle)@velocity / np.linalg.norm(velocity)
	side2 = -1*R(-angle*1.15)@velocity / np.linalg.norm(velocity)
	
	ax.plot([x_traj[-1]+scale*side2[0]+shift*dx,x_traj[-1]+shift*dx,x_traj[-1]+scale*side1[0]+shift*dx],[y_traj[-1]+scale*side2[1]+shift*dy,y_traj[-1]+shift*dy,y_traj[-1]+scale*side1[1]+shift*dy],'k',linewidth=linewidth)

def regularized_trajectory(bond,x0,y0,us):
	def dXdu(X,u):
		H = bond.hessian(X[0],X[1])
		return (adjugate(H)@ell).reshape(2,)

	Xs = odeint(dXdu,[x0,y0],us)
	return Xs


if 1: # Big Flow Diagram
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

	#plot_contours(selectin,xmin,xmax,ymin,ymax)

	print(selectin.xs,selectin.ys)
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

	#ax.plot(xss,yss)

	catch_to_slip_index = np.argmax(Ebs)

	H = selectin.hessian(switch_points[1][0],switch_points[1][1])
	lambdas,vs = np.linalg.eigh(H)
	v0 = vs[:,0]
	vscale = 0.01
	M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2]])
	Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index]]
	A = np.linalg.solve(M.T,Y)
	#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

	switchline_xs = np.linspace(xmin,switch_points[1][0],100)
	switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
	ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
	print('A = ',A)
	selectin.h_force = [0,0]
	selectin.f = 0
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)
	###
	###
	###


	for point in switch_points:
		plt.plot(point[0],point[1],'o',color='#ff7700',markersize=18)

	plt.plot([selectin.xm-.02],[selectin.ym+.02],'o',color='k',markersize=14)
	plt.plot([selectin.xs],[selectin.ys+.02],'X',color='k',markersize=14)

	ax.set_xlim(1.7*W,xmax)
	ax.set_ylim(-.1,.7)

	ax.set_xticks([1.7*W,1.8*W,1.9*W,xmax])
	ax.set_xticklabels(['']*4)
	ax.set_yticks([0.01,1/a])
	ax.set_yticklabels(['',''])

	plot_vector_field()


	ax.set_aspect(1.05, adjustable='box')

	plt.savefig('big_flow.eps',transparent=True)
	plt.show()
	quit()


fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(9,4.7),gridspec_kw={'height_ratios': [1, 0.55]})
CP_size = 10
SP_size = 14
SL_width = 3
traj_width = 2
Sepx_linewidth = 1.7
Sepx_linestyle = (0,(1,1))


if 0: # Slip-catch-slip # 1 
	#Parameters from L-selectin -- PSGL1
	W = 2.8
	a = 3.5 # Increasing this increases peak of catch
	sigma = 0.54 # Increasing this pulls initial point upwards, increases peak of catch
	D0 = 220  # Increasing seems to shift the energies up without doing much else
	k_theta = 350 # 266 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.6*pi
	theta1 = 1.*pi

	xmin,xmax,ymin,ymax = 4.,5.5999,-0.35,.8
	selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)

	plt.figure(figsize=(2.4,2.4))
	switch_points = plot_detH0_curve(selectin,xmin,xmax,ymin,ymax)
	#print('switch_points = ',switch_points)
	plt.plot([selectin.xm],[selectin.ym],'o',color='k')
	plt.plot([selectin.xs],[selectin.ys],'X',color='k')

	fs = np.linspace(0,80,50)
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	f20_index = int(20/max(fs) * 50)
	slip_to_catch_index = np.argmin(Ebs[0:f20_index])
	catch_to_slip_index = f20_index + np.argmax(Ebs[f20_index:])

	if 0: # Finding switch line (to 2nd order)
		M = np.array([[1,1,1],[switch_points[1][0],xss[catch_to_slip_index],xss[slip_to_catch_index]],[(switch_points[1][0])**2,(xss[catch_to_slip_index])**2,(xss[slip_to_catch_index])**2]])
		Y = [switch_points[1][1],yss[catch_to_slip_index],yss[slip_to_catch_index]]
		A = np.linalg.solve(M.T,Y)
		switchline_xs = np.linspace(xmin,xmax,100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
		plt.plot(switchline_xs,switchline_ys,color='#ff7700')
		print('A = ',A)
	
	if 1: # Finding switch line (to 3rd order)

		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		a = -0.1
		M = np.array([[1,1,1,1],[switch_points[1][0],switch_points[1][0]+a*v0[0],xss[catch_to_slip_index],xss[slip_to_catch_index]],[(switch_points[1][0])**2,(switch_points[1][0]+a*v0[0])**2,(xss[catch_to_slip_index])**2,(xss[slip_to_catch_index])**2],[(switch_points[1][0])**3,(switch_points[1][0]+a*v0[0])**3,(xss[catch_to_slip_index])**3,(xss[slip_to_catch_index])**3]])
		Y = [switch_points[1][1],switch_points[1][1]+a*v0[1],yss[catch_to_slip_index],yss[slip_to_catch_index]]
		A = np.linalg.solve(M.T,Y)
		#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

		switchline_xs = np.linspace(xmin,switch_points[1][0],100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2 + A[3]*switchline_xs**3
		plt.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=2)
		print('A = ',A)






	plt.plot(xms,yms,color='k')
	plt.plot(xss,yss,color='k')
	#plt.plot(xss[slip_to_catch_index],yss[slip_to_catch_index],'o')
	#plt.plot(xss[catch_to_slip_index],yss[catch_to_slip_index],'o')

	plt.ylim(-.3,1.15)
	plt.xlim(4.2,5.5999)

	plt.figure()
	plt.plot(fs,Ebs)
	plt.plot(fs[slip_to_catch_index],Ebs[slip_to_catch_index],'o')
	plt.plot(fs[catch_to_slip_index],Ebs[catch_to_slip_index],'o')

	#plt.show()


if 1: # Slip-catch-slip # 2 
	ax = axes[0,1]

	#Parameters from L-selectin -- PSGL1
	W = 2.7
	a = 2.9 #1/nm
	sigma = 0.5 # radians
	D0 = 220 # pNnm
	k_theta = 350 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.6*pi
	theta1 = 1.*pi

	xmin,xmax,ymin,ymax = 4.,2*W*0.99999,-0.35,1.5
	selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)


	switch_points, temp = plot_detH0_curve(ax,selectin,xmin,xmax,ymin,ymax)


	### Separatrix:
	###
	us = np.linspace(0,-.013,1000)
	X = regularized_trajectory(selectin,switch_points[0][0]+.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)
	
	us = np.linspace(0,.02,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	us = np.linspace(0,-.01,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]-.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	###
	###


	for point in switch_points:
		ax.plot(point[0],point[1],'o',color='#ff7700',markersize=SP_size)


	#ax.plot([selectin.xm],[selectin.ym],'o',color='k')
	#ax.plot([selectin.xs],[selectin.ys],'X',color='k')

	fs1 = np.linspace(0,15,50)
	fs2 = np.linspace(15,200,100)
	fs = np.concatenate((fs1,fs2))
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	f20_index = np.argmin((fs-20)**2)
	slip_to_catch_index = np.argmin(Ebs[0:f20_index])
	catch_to_slip_index = f20_index + np.argmax(Ebs[f20_index:])

	if 0: # Finding switch line (to 2nd order)
		M = np.array([[1,1,1],[switch_points[1][0],xss[catch_to_slip_index],xss[slip_to_catch_index]],[(switch_points[1][0])**2,(xss[catch_to_slip_index])**2,(xss[slip_to_catch_index])**2]])
		Y = [switch_points[1][1],yss[catch_to_slip_index],yss[slip_to_catch_index]]
		A = np.linalg.solve(M.T,Y)
		switchline_xs = np.linspace(xmin,xmax,100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
		plt.plot(switchline_xs,switchline_ys,color='#ff7700')
		print('A = ',A)
	
	if 1: # Finding switch line (to 3rd order)

		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		vscale = -0.1
		M = np.array([[1,1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index],xss[slip_to_catch_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2,(xss[slip_to_catch_index])**2],[(switch_points[1][0])**3,(switch_points[1][0]+vscale*v0[0])**3,(xss[catch_to_slip_index])**3,(xss[slip_to_catch_index])**3]])
		Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index],yss[slip_to_catch_index]]
		A = np.linalg.solve(M.T,Y)
		#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

		switchline_xs = np.linspace(xmin,switch_points[1][0],100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2 + A[3]*switchline_xs**3
		ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
		print('A = ',A)




	x_shift = .01
	start_index = 13
	ax.plot(xms[start_index:]+x_shift,yms[start_index:],color='k',linewidth=traj_width)
	ax.plot(xss[start_index:],yss[start_index:],color='k',linewidth=traj_width)
	ax.plot(xms[start_index]+x_shift,yms[start_index],'o',color='k',markersize=CP_size)
	ax.plot(xss[start_index],yss[start_index],'X',color='k',markersize=CP_size)


	plot_arrow(ax,xss,yss,scale=.07,linewidth=traj_width,shift=5)
	plot_arrow(ax,xms+x_shift,yms,scale=.07,linewidth=traj_width,shift=10)

	ax.set_xlim(1.54*W,1.999*W)
	ax.set_xticks([1.54*W,1.77*W,1.999*W])
	ax.set_xticklabels(['']*3)
	ax.set_ylim(-.2,1.01)
	ax.set_yticks([0,1/a])
	ax.set_yticklabels(['',''])



	ax = axes[1,1]
	ax.plot([fs[catch_to_slip_index],fs[catch_to_slip_index]],[0,100],color='#ff7700',linewidth=2)
	ax.plot([fs[slip_to_catch_index],fs[slip_to_catch_index]],[0,100],color='#ff7700',linewidth=2)
	ax.plot(fs,Ebs,'k')
	ax.set_ylim(30,34)
	ax.set_xlim(-2,80)

	ax.set_yticks([30,32,34])
	ax.set_yticklabels([30,'',34])
	ax.set_xticks([0,20,40,60,80])
	ax.set_xticklabels([0,'','','',80])



	#plt.show()



if 1: # Catch-slip
	ax = axes[0,0]


	#Parameters from L-selectin -- PSGL1
	W = 2.8
	a = 3  # 3. #1/nm
	sigma = 0.4 # radians
	D0 = 220 # pNnm
	k_theta = 350 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.6*pi
	theta1 = 1.*pi

	xmin,xmax,ymin,ymax = 4.7,5.5999,-0.2,0.6
	selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)

	switch_points,temp = plot_detH0_curve(ax,selectin,xmin,xmax,ymin,ymax)

	### Separatrix:
	###
	us = np.linspace(0,-.007,1000)
	X = regularized_trajectory(selectin,switch_points[0][0]+.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)
	
	us = np.linspace(0,.007,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	us = np.linspace(0,-.005,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]-.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	###
	###


	for point in switch_points:
		ax.plot(point[0],point[1],'o',color='#ff7700',markersize=SP_size)

	x_shift = -.013 # For visual clarity
	ax.plot([selectin.xm+x_shift],[selectin.ym+.005],'o',color='k',markersize=CP_size)
	ax.plot([selectin.xs+x_shift],[selectin.ys],'X',color='k',markersize=CP_size)




	fs1 = np.linspace(0,50,50)
	fs2 = np.linspace(50,230,50)
	fs = np.concatenate((fs1,fs2))

	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])
	selectin.f = 0

	catch_to_slip_index = np.argmax(Ebs)

	### Switch line:
	###

	if 0: # Finding switchline to 2nd order
		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		vscale = 0.01
		M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2]])
		Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index]]
		A = np.linalg.solve(M.T,Y)
		#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

		switchline_xs = np.linspace(xmin,switch_points[1][0],100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
		ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
		#print('A = ',A)


	if 0: # Finding switchline using CubicSpline
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
			bond.f = 0
			bond.h_force = [0,0]
			

			if switch_type=='catch_to_slip':
				switch_index = np.argmax(Ebs)
			elif switch_type=='slip_to_catch':
				switch_index = np.argmin(Ebs)

			x_switch,y_switch = xss[switch_index],yss[switch_index]
			plt.plot(xss,yss,'--',color='green')
			plt.plot(xss[0],yss[0],'o',color='green')
			plt.plot(x_switch,y_switch,'o',color='orange')
			return x_switch,y_switch


		x0,y0 = xss[catch_to_slip_index],yss[catch_to_slip_index]
		x1,y1 = find_switch(selectin,[30,-25],np.linspace(32,100,50),'catch_to_slip')

		grad1 = selectin.grad_V(5,0.6)
		print(grad1)
		x2,y2 = find_switch(selectin,grad1,np.linspace(-4.5,5,30),'slip_to_catch')

		Eb_switch_index = np.argmax(Ebs)
		x_switches = np.array([x0,x1,x2])
		y_switches = np.array([y0,y1,y2])


		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		vscale = 0.01

		x_switches = np.append(x_switches,[switch_points[1][0],switch_points[1][0]+vscale*v0[0]])
		y_switches = np.append(y_switches,[switch_points[1][1],switch_points[1][1]+vscale*v0[1]])

		reordered_indices = np.argsort(x_switches)
		x_switches = x_switches[reordered_indices]
		y_switches = y_switches[reordered_indices]

		from scipy.interpolate import CubicSpline
		switchline_xs = np.linspace(4.7,switch_points[1][0],100)
		switchline_func = CubicSpline(x_switches,y_switches)
		ax.plot(switchline_xs,switchline_func(switchline_xs),color='#ff7700',linewidth=SL_width)

	
	if 1: # Switch line from big flow
		selectin.h_force = [80,30]
		fs1 = np.linspace(0,50,50)
		num_fs1 = len(fs1)
		xms1,yms1,xss1,yss1 = np.zeros(num_fs1),np.zeros(num_fs1),np.zeros(num_fs1),np.zeros(num_fs1)
		Ebs1 = np.zeros(num_fs1)
		for i in range(num_fs1):
			selectin.f = fs[i]
			selectin.find_minimum()
			selectin.find_saddle(xmin,xmax,ymin,ymax)
			xms1[i],yms1[i],xss1[i],yss1[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
			Ebs1[i] = selectin.V(xss1[i],yss1[i]) - selectin.V(xms1[i],yms1[i])

		#ax.plot(xss,yss)

		catch_to_slip_index1 = np.argmax(Ebs1)

		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		vscale = 0.01
		M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss1[catch_to_slip_index1]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss1[catch_to_slip_index1])**2]])
		Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss1[catch_to_slip_index1]]
		A = np.linalg.solve(M.T,Y)
		#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

		switchline_xs = np.linspace(xmin,switch_points[1][0],100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
		ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
		print('A = ',A)
		selectin.h_force = [0,0]
		selectin.f = 0
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
	###
	###

	#Plot trajectories:
	ax.plot(xms+x_shift,yms,color='k',linewidth=traj_width)
	ax.plot(xss+x_shift,yss,color='k',linewidth=traj_width)

	plot_arrow(ax,xss+x_shift,yss,scale=.045,linewidth=traj_width,shift=1)
	plot_arrow(ax,xms+x_shift,yms,scale=.045,linewidth=traj_width,shift=2)

	ax.set_xlim(1.7*W,1.999*W)
	ax.set_xticks([1.7*W,1.8*W,1.9*W,1.999*W])
	ax.set_xticklabels(['','','',''])
	ax.set_ylim(-.1,.6)
	ax.set_yticks([0,1/a])
	ax.set_yticklabels(['',''])


	ax = axes[1,0]
	ax.plot(fs,Ebs,'k')
	ax.plot([fs[catch_to_slip_index],fs[catch_to_slip_index]],[0,100],color='#ff7700',linewidth=2)
	ax.set_ylim(10,30)
	ax.set_yticks([10,20,30])
	ax.set_yticklabels([10,'',30])

	ax.set_xlim(-4,200)
	ax.set_xticks([0,50,100,150,200])
	ax.set_xticklabels([0,'','','',200])

	#plt.show()


if 0: # Slip # 1
	ax = axes[0,2]

	#Parameters from L-selectin -- PSGL1
	W = 2.8
	a = 2.8 #1/nm
	sigma = 0.65 # radians
	D0 = 110 # 150 # pNnm
	k_theta = 120 # 190 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.6*pi
	theta1 = 1.*pi

	xmin,xmax,ymin,ymax = 3.2,5.5999,-0.2,1.5
	selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)

	print(selectin.xs,selectin.ys)
	switch_points , temp = plot_detH0_curve(ax,selectin,xmin,xmax,ymin,ymax)
	#ax.plot([selectin.xm],[selectin.ym],'o',color='k')
	#ax.plot([selectin.xs],[selectin.ys],'X',color='k')

	### Separatrix:
	###
	us = np.linspace(0,-.08,1000)
	X = regularized_trajectory(selectin,switch_points[0][0]+.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)
	
	us = np.linspace(0,.08,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	us = np.linspace(0,-.06,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]-.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	###
	###



	for point in switch_points:
		ax.plot(point[0],point[1],'o',color='#ff7700',markersize=SP_size)

	###
	### Finding switch line
	###
	selectin.h_force = [-10,30]
	fs = np.linspace(0,20,50)
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	#ax.plot(xss,yss)

	catch_to_slip_index = np.argmax(Ebs)

	H = selectin.hessian(switch_points[1][0],switch_points[1][1])
	lambdas,vs = np.linalg.eigh(H)
	v0 = vs[:,0]
	vscale = 0.01
	M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2]])
	Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index]]
	A = np.linalg.solve(M.T,Y)
	#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

	switchline_xs = np.linspace(xmin,switch_points[1][0],100)
	switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
	ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
	print('A = ',A)
	selectin.h_force = [0,0]
	###
	###
	###


	fs = np.linspace(0,123,150)
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	x_shift = .0
	start_index = 17
	ax.plot(xms[start_index:]+x_shift,yms[start_index:],color='k',linewidth=traj_width)
	ax.plot(xss[start_index:],yss[start_index:],color='k',linewidth=traj_width)
	ax.plot(xms[start_index]+x_shift,yms[start_index],'o',color='k',markersize=CP_size)
	ax.plot(xss[start_index],yss[start_index],'X',color='k',markersize=CP_size)


	plot_arrow_special(ax,xss+x_shift,yss,scale=.08,linewidth=traj_width,shift=1,angle=0.7)
	plot_arrow(ax,xms+x_shift,yms,scale=.08,linewidth=traj_width,shift=2,angle=0.8)

	ax.set_xlim(1.2*W,1.999*W)
	ax.set_xticks([1.2*W,1.4*W,1.6*W,1.8*W,1.999*W])
	ax.set_xticklabels(['']*5)
	ax.set_ylim(-.2,.95)
	ax.set_yticks([0,1/a])
	ax.set_yticklabels(['',''])

	ax = axes[1,2]
	ax.plot(fs,Ebs,'k')

	ax.set_xlim(-2,80)
	ax.set_ylim(20,50)

	ax.set_xticks([0,20,40,60,80])
	ax.set_xticklabels([0,'','','',80])
	ax.set_yticks([20,30,40,50])
	ax.set_yticklabels([20,'','',50])


if 1: # Slip # 2
	ax = axes[0,2]

	#Parameters from L-selectin -- PSGL1
	W = 2.8
	a = 2.8 #1/nm
	sigma = 0.53 # radians
	D0 = 110 # 150 # pNnm
	k_theta = 120 # 190 # pNnm
	gamma = 0.000033 # pN s / nm
	theta0 = 0.6*pi
	theta1 = 1.*pi

	xmin,xmax,ymin,ymax = 3.2,5.5999,-0.2,1.5
	selectin = Selectin(W=W,a=a,sigma=sigma,D0=D0,k_theta=k_theta,theta0=theta0,theta1=theta1,gamma=gamma)
	selectin.find_minimum()
	selectin.find_saddle(xmin,xmax,ymin,ymax)

	print(selectin.xs,selectin.ys)
	switch_points , temp = plot_detH0_curve(ax,selectin,xmin,xmax,ymin,ymax)
	#ax.plot([selectin.xm],[selectin.ym],'o',color='k')
	#ax.plot([selectin.xs],[selectin.ys],'X',color='k')

	### Separatrix:
	###
	us = np.linspace(0,-.08,1000)
	X = regularized_trajectory(selectin,switch_points[0][0]+.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)
	
	us = np.linspace(0,.08,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]+.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	us = np.linspace(0,-.06,100)
	X = regularized_trajectory(selectin,switch_points[0][0]-.01,switch_points[0][1]-.01,us)
	ax.plot(X[:,0],X[:,1],color='#5c5c5c',linewidth=Sepx_linewidth,linestyle=Sepx_linestyle)

	###
	###



	for point in switch_points:
		ax.plot(point[0],point[1],'o',color='#ff7700',markersize=SP_size)

	###
	### Finding switch line
	###
	selectin.h_force = [0,5]
	fs = np.linspace(0,20,50)
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	#ax.plot(xss,yss)

	catch_to_slip_index = np.argmax(Ebs)

	H = selectin.hessian(switch_points[1][0],switch_points[1][1])
	lambdas,vs = np.linalg.eigh(H)
	v0 = vs[:,0]
	vscale = 0.01
	M = np.array([[1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2]])
	Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index]]
	A = np.linalg.solve(M.T,Y)
	#plt.plot([switch_points[1][0]+0.1*v0[0],switch_points[1][0]-0.1*v0[0]],[switch_points[1][1]+0.1*v0[1],switch_points[1][1]-0.1*v0[1]])

	switchline_xs = np.linspace(xmin,switch_points[1][0],100)
	switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2
	ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
	print('A = ',A)
	selectin.h_force = [0,0]
	###
	###
	###


	fs = np.linspace(0,123,150)
	num_fs = len(fs)
	xms,yms,xss,yss = np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs),np.zeros(num_fs)
	Ebs = np.zeros(num_fs)
	for i in range(num_fs):
		selectin.f = fs[i]
		selectin.find_minimum()
		selectin.find_saddle(xmin,xmax,ymin,ymax)
		xms[i],yms[i],xss[i],yss[i] = selectin.xm,selectin.ym,selectin.xs,selectin.ys
		Ebs[i] = selectin.V(xss[i],yss[i]) - selectin.V(xms[i],yms[i])

	x_shift = .0
	start_index = 4
	ax.plot(xms[start_index:]+x_shift,yms[start_index:],color='k',linewidth=traj_width)
	ax.plot(xss[start_index:],yss[start_index:],color='k',linewidth=traj_width)
	ax.plot(xms[start_index]+x_shift,yms[start_index],'o',color='k',markersize=CP_size)
	ax.plot(xss[start_index],yss[start_index],'X',color='k',markersize=CP_size)


	plot_arrow_special(ax,xss+x_shift,yss,scale=.077,linewidth=traj_width,shift=2,angle=0.55)
	plot_arrow(ax,xms+x_shift,yms,scale=.08,linewidth=traj_width,shift=2,angle=0.6)

	ax.set_xlim(1.4*W,1.999*W)
	ax.set_xticks([1.4*W,1.6*W,1.8*W,1.999*W])
	ax.set_xticklabels(['']*4)
	ax.set_ylim(-.2,1.02)
	ax.set_yticks([0,1/a])
	ax.set_yticklabels(['',''])

	ax = axes[1,2]
	ax.plot(fs,Ebs,'k')

	ax.set_xlim(-2,80)
	ax.set_ylim(20,40)

	ax.set_xticks([0,20,40,60,80])
	ax.set_xticklabels([0,'','','',80])
	ax.set_yticks([20,30,40])
	ax.set_yticklabels([20,'',40])




for ax in axes[1,:]:
	ax.tick_params(pad=2,length=3)

plt.subplots_adjust(hspace=.32,wspace=0.25)
plt.savefig('fig1_EFG.eps',transparent=True)
plt.show()




#