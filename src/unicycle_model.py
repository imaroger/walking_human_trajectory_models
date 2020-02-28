import numpy as np
import matplotlib.pylab as plt
import crocoddyl

model = crocoddyl.ActionModelUnicycle()
data  = model.createData()

model.costWeights = np.matrix([
    10,   # state weight
    10    # control weight
]).T

x0 = np.matrix([ -1, 1, 3.14 ]).T #x,y,theta
T  = 50
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model)

ddp = crocoddyl.SolverDDP(problem)
done = ddp.solve()
assert(done)

X = []
Y = []

sc = .1
count = 0
for state in ddp.xs:
	x, y, th = np.asscalar(state[0]), np.asscalar(state[1]), np.asscalar(state[2])		
	if count%5 == 0:
		c, s = np.cos(th), np.sin(th)	
		plt.arrow(x, y, c * sc, s * sc, head_width=.05)
	X.append(x)
	Y.append(y)
	count += 1
plt.axis([-2, 2., -2., 2.])
plt.show()

plt.plot(X,Y)
plt.axis([-2,2,-2,2])
plt.grid(True)
plt.show()

print(ddp.xs[-1])

