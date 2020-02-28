import numpy as np
from scipy.integrate import odeint
from math import pi, floor
import matplotlib.pylab as plt
from math import atan2

#############################################################################################
########################## BUILD THE CLOTHOID - FUNCTION DEFINITION	#########################
#############################################################################################

def clothoid_ode_rhs(state, s, kappa0, kappa1, alpha):
    x, y, theta = state[0], state[1], state[2]
    return np.array([alpha*np.cos(theta), alpha*np.sin(theta), kappa0 + kappa1*s])

def eval_clothoid(x0,y0,theta0, kappa0, kappa1, s, alpha):
    return odeint(clothoid_ode_rhs, np.array([x0,y0,theta0]), s, (kappa0, kappa1, alpha))  

def generalized_fresnel_fct(state,t,n,k,dk,theta0,alpha):
	return np.array([alpha*t**n*np.cos(theta0+k*t+dk*t**2/2),alpha*t**n*np.sin(theta0+k*t+dk*t**2/2)]) 

def int_generalized_fresnel_fct(n,k,dk,theta0,alpha):
	sol = [[],[]]
	t = [0,1]
	for i in range (0,n):
		sol_i = odeint(generalized_fresnel_fct, np.array([0,0]), t, (i, k, dk, theta0, alpha))
		sol[0].append(sol_i[1,0])
		sol[1].append(sol_i[1,1])		
	return sol

def normalizeAngle(angle): 
	new_angle = angle
	while new_angle > pi:
		new_angle -= 2*pi
	while new_angle < -pi:
		new_angle += 2*pi
	return new_angle	

def build_clothoid_param(x0,y0,theta0,xf,yf,thetaf,alpha):
	dx = xf-x0
	dy = yf-y0
	R = np.sqrt(dx**2+dy**2)
	phi = atan2(dy,dx)
	phi0 =normalizeAngle(theta0-phi)
	phif = normalizeAngle(thetaf-phi)
	dphi = phif-phi0

	A_guess = guessA(phi0,phif)

	epsilon = 1e-12

	A = findA(A_guess,dphi,phi0,epsilon,alpha)

	sol = int_generalized_fresnel_fct(1,dphi-A,2*A,phi0,alpha)
	h = sol[0][0]
	L = R/h

	if L > 0:
		k = (dphi-A)/L
		dk = 2*A/L**2
		return (k,dk,L)
	else :
		print ("error: negative length")
		return(0,0,0)

def guessA(phi0,phif):
	coef = [2.989696028701907,0.716228953608281,-0.458969738821509,
	-0.502821153340377,0.261062141752652,-0.045854475238709]
	X  = phi0/pi
	Y  = phif/pi
	xy = X*Y
	A  = (phi0+phif) * ( coef[0] + xy * ( coef[1] + xy * coef[2] )+
		(coef[3] + xy * coef[4]) * (X**2+Y**2) + coef[5] * (X**4+Y**4))
	return A

def findA(A_guess, dphi, phi0, epsilon, alpha):
	A = A_guess 
	k = 1
	n = 0
	while abs(k) > epsilon and n < 100:
		a = dphi-A
		b = 2*A
		c = phi0
		sol = int_generalized_fresnel_fct(3,a,b,c,alpha)		
		intS = sol[1]
		intC = sol[0]
		k = intS[0]
		dk = intC[2]-intC[1]
		A  = A - k/dk
		n += 1
	if abs(k) > epsilon*10:
		print("Newton iteration fails, k = ",k)
		return 0
	else:
		return A

def plotClothoid(clothoid, title):
	x, y, theta = clothoid[:,0], clothoid[:,1], clothoid[:,2] 
	plt.plot(x, y, lw=1)
	plt.title(title)

	arrow_len = 0.08
	count = 0
	for i in range (0,len(x)):
		if count%100 == 0:
			c, s = np.cos(theta[i]), np.sin(theta[i])	
			plt.arrow(x[i], y[i], c * arrow_len, s * arrow_len, head_width=.05)
		count += 1
	plt.arrow(x[-1], y[-1], np.cos(theta[-1]) * arrow_len, np.sin(theta[-1]) * arrow_len)
	

def writeClothoid(x0,y0,theta0,xf,yf,thetaf,step,alpha):
	theta0, thetaf = floor(theta0*100)/100, floor(thetaf*100)/100
	k,dk,L = build_clothoid_param(x0,y0,theta0,xf,yf,thetaf,alpha)
	print(L)
	sol = eval_clothoid(x0, y0, theta0, k, dk, np.arange(0,L,step), alpha)
	title = "Clothoid_from_"+str(x0)+","+str(y0)+","+str(theta0) \
	+"_to_"+str(xf)+","+str(yf)+","+str(thetaf)+"_"+str(step)
	print(title)
	plotClothoid(sol,title)
	#plt.show()
	path = "../data/Clothoid/"+title+"_pos.dat"
	np.savetxt(path,sol)
	vsol = np.zeros((len(sol[:,0])-1,3))
	for i in range (0,len(sol[:,0])-1):
		vsol[i][0]=(sol[:,0][i+1]-sol[:,0][i])/step
		vsol[i][1]=(sol[:,1][i+1]-sol[:,1][i])/step		
		vsol[i][2]=(sol[:,2][i+1]-sol[:,2][i])/step
	vpath = "../data/Clothoid/"+title+"_vel.dat"
	np.savetxt(vpath,vsol)

###############################################################################
################################## MAIN #######################################
###############################################################################

writeClothoid(0, 0, 0, 2, 1, pi/2, 0.01, 0.1)
plt.show()
