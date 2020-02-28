import numpy as np
import matplotlib.pylab as plt
import crocoddyl
from math import pi, floor

#########################################################################
########################## FUNCTION DEFINITION	#########################
#########################################################################


def costFunction(weights,x,u,posf):
	return weights[0] + weights[1]*u[-1][0]**2 + weights[2]*u[-1][1]**2 + \
		weights[3]*u[-1][2]**2 + weights[4]*(np.arctan2(posf[1]-x[-1][1], \
		posf[0]-x[-1][0]) - x[-1][2])**2 + weights[5]*((posf[0]-x[-1][0])**2 + \
		(posf[1]-x[-1][1])**2) + weights[6]*(posf[2]-x[-1][2])**2


def optimizeT(T_min,T_max,X0):
	T_opt  =  T_min
	cost_min = - 1
	for T in range (T_min,T_max,5):
		problem = crocoddyl.ShootingProblem(X0, [ model ] * T, model)
		ddp = crocoddyl.SolverDDP(problem)
		done = ddp.solve()
		cost = costFunction(model.costWeights,ddp.xs,ddp.us,model.finalState)
		if (cost_min < 0 or cost_min > abs(cost) and ddp.iter < 80):
			cost_min = abs(cost)
			T_opt = T
	return T_opt	

def plotOptControl(xs):
	X,Y = [],[]
	arrow_len = 0.08
	count = 0
	for state in xs:
		x, y, th = np.asscalar(state[0]), np.asscalar(state[1]), np.asscalar(state[2])		
		if count%10 == 0:
			c, s = np.cos(th), np.sin(th)	
			plt.arrow(x, y, c * arrow_len, s * arrow_len, head_width=.05)
		X.append(x)
		Y.append(y)
		count += 1
	plt.arrow(x, y, np.cos(th) * arrow_len, np.sin(th) * arrow_len)
	plt.plot(X,Y)

def plotOptControlTranslate(xs,posf):
	X,Y = [],[]
	arrow_len = 0.05
	count = 0
	for state in xs:
		x, y, th = np.asscalar(state[0])-posf[0], np.asscalar(state[1])-posf[1], np.asscalar(state[2])		
		if count%30 == 0:
			c, s = np.cos(th), np.sin(th)	
			plt.arrow(x, y, c * arrow_len, s * arrow_len, head_width=.03)
		X.append(x)
		Y.append(y)
		count += 1
	plt.arrow(x, y, np.cos(th) * arrow_len, np.sin(th) * arrow_len)
	plt.plot(X,Y)

def writeOptControl(X0,xf,yf,thetaf,T_opt):
	model.finalState = np.matrix([xf,yf,thetaf]).T
	problem = crocoddyl.ShootingProblem(X0, [ model ] * T_opt, model)
	ddp = crocoddyl.SolverDDP(problem)
	done = ddp.solve()
	title = "OptControl_from_"+str(-xf)+","+str(-yf)+","+str(floor((X0[2])*100)/100)+"_to_0,0,"+str(floor(thetaf*100)/100)
	sol = np.transpose(ddp.xs)[0]
	print(title)
	plotOptControlTranslate(ddp.xs, [xf,yf,thetaf])
	sol[0] = np.array(sol[0])-xf
	sol[1] = np.array(sol[1])-yf
	plt.title(title)
	plt.plot(sol[0],sol[1])
	path = "../data/OptControl/"+title+"_pos.dat"
	np.savetxt(path,np.transpose(sol))
	vsol = np.zeros((len(sol[0])-1,3))
	for i in range (0,len(sol[0])-1):
		vsol[i][0]=(sol[0][i+1]-sol[0][i])/0.01
		vsol[i][1]=(sol[1][i+1]-sol[1][i])/0.01	
		vsol[i][2]=(sol[2][i+1]-sol[2][i])/0.01
	vpath = "../data/OptControl/"+title+"_vel.dat"
	np.savetxt(vpath,vsol)


########################################################################
################################## MAIN ################################
########################################################################


model = crocoddyl.ActionModelHuman()
data  = model.createData()

T_min = 250
T_max = 500

model.costWeights = np.matrix([1.,1.2,1.7,.7,5.2,5,8]).T
model.alpha = 1

pos_f = [1.,2.,pi/2]
model.finalState = np.matrix(pos_f).T

model.alpha = 1.
X0 = np.matrix([ 0., 0., pi/2, 0., 0., 0.]).T #x,y,theta,vf,w,vo
T_opt = optimizeT(T_min, T_max,X0)
writeOptControl(X0, pos_f[0], pos_f[1], pos_f[2], T_opt)

plt.show()
