import numpy as np
import matplotlib.pyplot as plt
import types

class Cubic_bond_1D:
	def __init__(self,a,b,c,ell=0,f=0):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.ell = ell
		self.f = f

	def V(self,x):
		return self.a*x +self.b*x**2 + self.c*x**3 - self.f*self.ell*x

class Simple_cubic_2D:
	def __init__(self,a,b,c,d,ell=0,f=0,T=4.28):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.ell = ell
		self.f = f
		self.T = T
		self.xm = None #x_minimum
		self.ym = None #y_minimum
		self.xs = None #x_saddle
		self.ys = None #y_saddle

	def V(self,x,y):
		return self.a*x + self.b*x**2 + self.c*x**3 + self.d*y**2 - self.f*self.ell*x

	def find_critical_points(self):
		self.ym = 0
		self.ys = 0

		if self.c>0:
			self.xm = (-2*self.b + np.sqrt(4*self.b**2 - 12*(self.a-self.f*self.ell)*self.c))/(6*self.c)
			self.xs = (-2*self.b - np.sqrt(4*self.b**2 - 12*(self.a-self.f*self.ell)*self.c))/(6*self.c)
		else:
			self.xm = (-2*self.b - np.sqrt(4*self.b**2 - 12*(self.a-self.f*self.ell)*self.c))/(6*self.c)
			self.xs = (-2*self.b + np.sqrt(4*self.b**2 - 12*(self.a-self.f*self.ell)*self.c))/(6*self.c)			

	def find_minimum(self):
		self.find_critical_points()

	def force(self,x,y):
		#Computing force at point x,y (due to bond and extenral applied force)
		h = 0.00001
		V0 = self.V(x,y)
		V_x = (self.V(x+h,y) - V0)/h
		V_y = (self.V(x,y+h) - V0)/h
		return -1*V_x,-1*V_y

	def hessian(self,x, y):
		h = 0.00001
		V0 = self.V(x, y)
		V_xx = (self.V(x + h, y) + self.V(x - h, y) - 2 * V0) / h**2
		V_yy = (self.V(x, y + h) + self.V(x, y - h) - 2 * V0) / h**2
		V_xy = (self.V(x + h, y + h) + V0 - self.V(x + h, y) - self.V(x, y + h)) / h**2
		return np.array([[V_xx, V_xy], [V_xy, V_yy]])

	def detH(self,x, y):
		h = 0.00001
		V0 = self.V(x, y)
		V_xx = (self.V(x + h, y) + self.V(x - h, y) - 2 * V0) / h**2
		V_yy = (self.V(x, y + h) + self.V(x, y - h) - 2 * V0) / h**2
		V_xy = (self.V(x + h, y + h) + V0 - self.V(x + h, y) - self.V(x, y + h)) / h**2
		return V_xx*V_yy-V_xy*V_xy

	def langers_rate(self):
		if self.xm==None or self.xs==None or self.ym==None or self.ys==None:
			self.find_critical_points()

		Eb = self.V(self.xs,self.ys) - self.V(self.xm,self.ym)
		H_saddle = self.hessian(self.xs,self.ys)
		H_evals = np.linalg.eigvalsh(H_saddle)
		omega_saddle_sqr = -1*min(H_evals)
		return np.exp(-Eb/self.T) * omega_saddle_sqr/(2*np.pi) * np.sqrt(-self.detH(self.xm,self.ym)/self.detH(self.xs,self.ys))



class Selectin:
	def __init__(self,W,theta0,sigma,D0,k_theta,a,theta1,gamma,f=0,T=4.28):
		self.W = W
		self.theta0 = theta0
		self.sigma = sigma
		self.D0 = D0
		self.k_theta = k_theta
		self.a = a
		self.theta1 = theta1
		self.gamma = gamma
		self.f = f
		self.h_force = [0,0]
		self.T = T
		#self.fitted_D = False
		self.xm = None #x_minimum
		self.ym = None #y_minimum
		self.xs = None #x_saddle
		self.ys = None #y_saddle

	def D(self,theta):
		return self.D0*np.exp(-0.5*(theta-self.theta1)**2/self.sigma**2)

	def D_adjustment(self,theta):
		return theta*0

	def modify_D_adjustment(self,new_func):
		#Allows you to insert your own custom function (new_func) into D_adjustment
		self.D_adjustment = types.MethodType(new_func,self)


	def V(self,x,y):
		#x: Extension E (E=2Wsin(theta/2))
		#y: Bond distance d
		
		theta = 2*np.arcsin(x/(2*self.W))
		V_theta = 0.5*self.k_theta*(theta-self.theta0)**2
		#if self.fitted_D:
		#	Ds = self.D_fitted(theta)
		#else:
		#	Ds = self.D(theta)
		Ds = self.D(theta) + self.D_adjustment(theta)
		M = (1-np.exp(-self.a*y))**2 - 1 
		return V_theta + Ds*M - self.f*(x+y) - self.h_force[0]*x - self.h_force[1]*y

	def force(self,x,y):
		#Computing force at point x,y (due to bond and extenral applied force)
		h = 0.00001
		V0 = self.V(x,y)
		V_x = (self.V(x+h,y) - V0)/h
		V_y = (self.V(x,y+h) - V0)/h
		return -1*V_x,-1*V_y

	def grad_V(self,x,y):
		#Computing force at point x,y (due to bond and extenral applied force)
		h = 0.00001
		V0 = self.V(x,y)
		V_x = (self.V(x+h,y) - V0)/h
		V_y = (self.V(x,y+h) - V0)/h
		return np.array([V_x,V_y])

	def grad_V_magnitude(self,x,y):
		h = 0.00001
		V0 = self.V(x,y)
		V_x = (self.V(x+h,y) - V0)/h
		V_y = (self.V(x,y+h) - V0)/h
		return V_x**2 + V_y**2

	def hessian(self,x, y):
		h = 0.00001
		V0 = self.V(x, y)
		V_xx = (self.V(x + h, y) + self.V(x - h, y) - 2 * V0) / h**2
		V_yy = (self.V(x, y + h) + self.V(x, y - h) - 2 * V0) / h**2
		V_xy = (self.V(x + h, y + h) + V0 - self.V(x + h, y) - self.V(x, y + h)) / h**2
		return np.array([[V_xx, V_xy], [V_xy, V_yy]])

	def detH(self,x, y):
		h = 0.00001
		V0 = self.V(x, y)
		V_xx = (self.V(x + h, y) + self.V(x - h, y) - 2 * V0) / h**2
		V_yy = (self.V(x, y + h) + self.V(x, y - h) - 2 * V0) / h**2
		V_xy = (self.V(x + h, y + h) + V0 - self.V(x + h, y) - self.V(x, y + h)) / h**2
		return V_xx*V_yy-V_xy*V_xy

	def langers_rate(self):
		if self.xm==None or self.xs==None or self.ym==None or self.ys==None:
			self.find_critical_points()

		Eb = self.V(self.xs,self.ys) - self.V(self.xm,self.ym)
		H_saddle = self.hessian(self.xs,self.ys)
		H_evals = np.linalg.eigvalsh(H_saddle)
		omega_saddle_sqr = -1*min(H_evals)
		return np.exp(-Eb/self.T) * omega_saddle_sqr/(2*np.pi) * np.sqrt(-self.detH(self.xm,self.ym)/self.detH(self.xs,self.ys))

	def find_minimum(self):
		Emin,Emax = 2*self.W*np.sin(self.theta0/2) * 0.99 , 2*self.W * 0.99999
		xs = np.linspace(Emin,Emax,100)
		ys = np.linspace(-.01,2/self.a,100)
		x_mesh,y_mesh = np.meshgrid(xs,ys)
		Vs = self.V(x_mesh,y_mesh)

		detHs = self.detH(x_mesh,y_mesh)
		Vs[detHs<=0] = np.inf
		min_y_index,min_x_index = np.unravel_index(Vs.argmin(), Vs.shape)
		min_x,min_y = xs[min_x_index],ys[min_y_index]
		if 1: #Adjust
			H = self.hessian(min_x,min_y)
			vect = np.array([min_x,min_y])
			min_vect = np.linalg.solve(H,H@vect - self.grad_V(min_x,min_y))
		else:
			min_vect = [min_x,min_y]
		self.xm , self.ym = np.min([min_vect[0],Emax]) , min_vect[1]


	def find_saddle(self,xmin,xmax,ymin,ymax):
		xs = np.linspace(xmin,xmax,100)
		ys = np.linspace(ymin,ymax,100)
		x_mesh,y_mesh = np.meshgrid(xs,ys)
		detHs = self.detH(x_mesh,y_mesh)
		grad_V_magnitudes = self.grad_V_magnitude(x_mesh,y_mesh)
		grad_V_magnitudes[detHs>=0] = np.inf

		#plt.figure()
		#plt.imshow(grad_V_magnitudes)
		#plt.show()

		saddle_y_index,saddle_x_index = np.unravel_index(grad_V_magnitudes.argmin(), grad_V_magnitudes.shape)
		saddle_x,saddle_y = xs[saddle_x_index],ys[saddle_y_index]
		if 1: #Adjust
			H = self.hessian(saddle_x,saddle_y)
			vect = np.array([saddle_x,saddle_y])
			saddle_vect = np.linalg.solve(H,H@vect - self.grad_V(saddle_x,saddle_y))
		else:
			saddle_vect = [saddle_x,saddle_y]
		self.xs , self.ys = saddle_vect[0] , saddle_vect[1]

	def find_saddle_precision(self,xmin,xmax,ymin,ymax,n_iter=3):
		xs = np.linspace(xmin,xmax,100)
		ys = np.linspace(ymin,ymax,100)
		x_mesh,y_mesh = np.meshgrid(xs,ys)
		detHs = self.detH(x_mesh,y_mesh)
		grad_V_magnitudes = self.grad_V_magnitude(x_mesh,y_mesh)
		grad_V_magnitudes[detHs>=0] = np.inf

		saddle_y_index,saddle_x_index = np.unravel_index(grad_V_magnitudes.argmin(), grad_V_magnitudes.shape)
		saddle_x,saddle_y = xs[saddle_x_index],ys[saddle_y_index]
		for i in range(n_iter):
			H = self.hessian(saddle_x,saddle_y)
			vect = np.array([saddle_x,saddle_y])
			saddle_vect = np.linalg.solve(H,H@vect - self.grad_V(saddle_x,saddle_y))
			saddle_x,saddle_y = saddle_vect[0],saddle_vect[1]
		self.xs , self.ys = saddle_vect[0] , saddle_vect[1]


	def langers_tau(self,xmin,xmax,ymin,ymax):
		#print('Inside langers_tau')
		self.find_minimum()
		self.find_saddle(xmin,xmax,ymin,ymax)
		
		#print(self.xm,self.ym,self.xs,self.ys)

		Eb = self.V(self.xs,self.ys) - self.V(self.xm,self.ym)
		#print(Eb)
		detA = self.detH(self.xm,self.ym)
		S_evals = np.linalg.eigvalsh(self.hessian(self.xs,self.ys))
		omega_S_stable = np.sqrt(max(S_evals))
		omega_S_unstable = np.sqrt(-min(S_evals))

		#print('prefactor = ',sqrt(detA)*omega_S_unstable/(2*pi*omega_S_stable),Eb)
		#print('Leaving langers_tau')
		return 1/(np.sqrt(detA)*omega_S_unstable/(2*np.pi*self.gamma*omega_S_stable) * np.exp(-Eb/self.T)) , Eb


class Tuned_Selectin(Selectin):
	def __init__(self,W,theta0,sigma,D0,k_theta,a,theta1,gamma,f=0,T=4.28,a1=0,mu1=0,s1=0,a2=0,mu2=0,s2=0,const=0):
		
		self.a1 = a1
		self.mu1 = mu1
		self.s1 = s1
		self.a2 = a2
		self.mu2 = mu2
		self.s2 = s2
		self.const = const
		super().__init__(W,theta0,sigma,D0,k_theta,a,theta1,gamma,f,T)

	def V(self,x,y):
		#x: Extension E (E=2Wsin(theta/2))
		#y: Bond distance d
		
		theta = 2*np.arcsin(x/(2*self.W))
		#weird = np.maximum(theta,self.theta0)
		V_theta = 0.5*self.k_theta*(theta-self.theta0)**2 #/ (1+0.01*theta)
		Ds = self.D0*np.exp(-0.5*(theta-self.theta1)**2/self.sigma**2) + self.const + self.a1*np.exp(-0.5*(x-self.mu1)**2/self.s1**2) + self.a2*np.exp(-0.5*(x-self.mu2)**2/self.s2**2)
		M = (1-np.exp(-self.a*y))**2 - 1 
		return V_theta + Ds*M - self.f*(x+y) #+ .1/(y+1)**12









