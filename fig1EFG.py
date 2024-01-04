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



fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(9,4.7),gridspec_kw={'height_ratios': [1, 0.55]})
CP_size = 10
SP_size = 14
SL_width = 3
traj_width = 2
Sepx_linewidth = 1.7
Sepx_linestyle = (0,(1,1))



if 1: # Slip-catch-slip 
	ax = axes[0,1]

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

	
	if 1: # Finding switch line

		H = selectin.hessian(switch_points[1][0],switch_points[1][1])
		lambdas,vs = np.linalg.eigh(H)
		v0 = vs[:,0]
		vscale = -0.1
		M = np.array([[1,1,1,1],[switch_points[1][0],switch_points[1][0]+vscale*v0[0],xss[catch_to_slip_index],xss[slip_to_catch_index]],[(switch_points[1][0])**2,(switch_points[1][0]+vscale*v0[0])**2,(xss[catch_to_slip_index])**2,(xss[slip_to_catch_index])**2],[(switch_points[1][0])**3,(switch_points[1][0]+vscale*v0[0])**3,(xss[catch_to_slip_index])**3,(xss[slip_to_catch_index])**3]])
		Y = [switch_points[1][1],switch_points[1][1]+vscale*v0[1],yss[catch_to_slip_index],yss[slip_to_catch_index]]
		A = np.linalg.solve(M.T,Y)
		switchline_xs = np.linspace(xmin,switch_points[1][0],100)
		switchline_ys = A[0] + A[1]*switchline_xs + A[2]*switchline_xs**2 + A[3]*switchline_xs**3
		ax.plot(switchline_xs,switchline_ys,color='#ff7700',linewidth=SL_width)
		print('A = ',A)



	start_index = 12
	ax.plot(xms[start_index:],yms[start_index:],color='k',linewidth=traj_width)
	ax.plot(xss[start_index:],yss[start_index:],color='k',linewidth=traj_width)
	ax.plot(xms[start_index],yms[start_index],'o',color='k',markersize=CP_size)
	ax.plot(xss[start_index],yss[start_index],'X',color='k',markersize=CP_size)


	plot_arrow(ax,xss,yss,scale=.07,linewidth=traj_width,shift=5)
	plot_arrow(ax,xms,yms,scale=.07,linewidth=traj_width,shift=10)

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


	ax.plot([selectin.xm],[selectin.ym+.005],'o',color='k',markersize=CP_size)
	ax.plot([selectin.xs],[selectin.ys],'X',color='k',markersize=CP_size)




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

	
	if 1: # Switch line
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


	#Plot trajectories:
	ax.plot(xms,yms,color='k',linewidth=traj_width)
	ax.plot(xss,yss,color='k',linewidth=traj_width)

	plot_arrow(ax,xss,yss,scale=.045,linewidth=traj_width,shift=1)
	plot_arrow(ax,xms,yms,scale=.045,linewidth=traj_width,shift=2)

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



if 1: # Slip
	ax = axes[0,2]

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


	if 1: # Switch line
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

	start_index = 4
	ax.plot(xms[start_index:],yms[start_index:],color='k',linewidth=traj_width)
	ax.plot(xss[start_index:],yss[start_index:],color='k',linewidth=traj_width)
	ax.plot(xms[start_index],yms[start_index],'o',color='k',markersize=CP_size)
	ax.plot(xss[start_index],yss[start_index],'X',color='k',markersize=CP_size)


	plot_arrow_special(ax,xss,yss,scale=.077,linewidth=traj_width,shift=2,angle=0.55)
	plot_arrow(ax,xms,yms,scale=.08,linewidth=traj_width,shift=2,angle=0.6)

	ax.set_xlim(1.4*W,1.999*W)
	ax.set_xticks([1.4*W,1.6*W,1.8*W,1.999*W])
	ax.set_xticklabels(['']*4)
	ax.set_ylim(-.2,1.01)
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
plt.show()




#