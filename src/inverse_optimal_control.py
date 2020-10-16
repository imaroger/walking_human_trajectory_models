import numpy as np
import matplotlib.pylab as plt
import crocoddyl
from math import pi, floor, sqrt, cos, sin, atan2
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import time


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#########################################################################
########################## FUNCTION DEFINITION	#########################
#########################################################################

def costFunction(weights,terminal_weights,x,u,posf):
	sum = 0
	for i in range (len(x)-1):
		sum += weights[0] + weights[1]*u[i][0]**2 + weights[2]*u[i][1]**2 + \
		weights[3]*u[i][2]**2 + weights[4]*(np.arctan2(posf[1]-x[i][1], \
		posf[0]-x[i][0]) - x[i][2])**2 
	sum += terminal_weights[0]*((posf[0]-x[-1][0])**2 + (posf[1]-x[-1][1])**2) +\
	terminal_weights[1]*(normalizeAngle(posf[2]-x[-1][2]))**2 +\
	terminal_weights[2]*(x[-1][3]**2 + x[-1][4]**2) +\
	terminal_weights[3]*x[-1][5]**2
	return sum

def optimizeT(T,x0,T_guess):
	T = int(T[0])
	if T > 0 and T < 2*T_guess:
		problem = crocoddyl.ShootingProblem(x0, [ model ] * T, terminal_model)
		ddp = crocoddyl.SolverDDP(problem)
		done = ddp.solve()
		# print(T,done,ddp.iter)
		if done:
			cost = costFunction(model.costWeights,terminal_model.costWeights,ddp.xs,ddp.us,model.finalState)/T
			# print(cost)
		elif ddp.iter < 50:
			cost = costFunction(model.costWeights,terminal_model.costWeights,ddp.xs,ddp.us,model.finalState)/T*10
		else:
			cost = 1e4
	else:
		cost = 1e8		
	return cost	

def translate(xs,x0):
	x,y,th = [],[],[]
	for state in xs:
		x.append(x0[0] + state[0])
		y.append(x0[1] + state[1])
		th.append(state[2])
	return (x,y,th)

def plotDdpResult(x,y,theta):
	arrow_len = 0.08
	count = 0
	for i in range (len(x)):
		if count%10 == 0:
			c, s = np.cos(theta[i]), np.sin(theta[i])	
			plt.arrow(x[i], y[i], c * arrow_len, s * arrow_len, head_width=.05)
		count += 1
	plt.plot(x,y)
	#plt.grid(True)

def solveDdp(pos_i,pos_f,graph_disp):
	# print("--- Ddp ---")
	final_state = [(pos_f[0]-pos_i[0]),(pos_f[1]-pos_i[1]),pos_f[2]]
	model.finalState = np.matrix([final_state[0],final_state[1],final_state[2]]).T
	terminal_model.finalState = np.matrix([final_state[0],final_state[1],final_state[2]]).T
	init_state = np.matrix([ 0, 0,pos_i[2] , 0, 0, 0]).T

	distance = sqrt((pos_f[0]-pos_i[0])**2+(pos_f[1]-pos_i[1])**2)
	T_guess = int(distance*100/model.alpha*2/3)

	optimal = minimize(optimizeT, T_guess, args=(init_state,distance*100/model.alpha),\
	method='Nelder-Mead',options = {'xtol': 0.01,'ftol': 0.001})
	T_opt = int(optimal.x)
	# print("----",T_guess,T_opt)

	problem = crocoddyl.ShootingProblem(init_state, [ model ] * T_opt, terminal_model)
	ddp = crocoddyl.SolverDDP(problem)
	done = ddp.solve()
	title = "DdpResult_from_"+str(pos_i[0])+","+str(pos_i[1])+","+\
	str(floor(pos_i[2]*100)/100)+"_to_0,0,"+str(floor(pos_f[2]*100)/100)
	x,y,theta = translate(ddp.xs, pos_i)

	if graph_disp:
		plotDdpResult(x,y,theta)

	path = "data/DdpResult/"+title+"_pos.dat"
	sol = [x,y,theta]
	np.savetxt(path,np.transpose(sol))
	# print("--- End of Ddp ---")

def computeMultipleTraj(list_pos_i,list_pos_f,graph_disp):
	print("Compute them all")
	count = 1
	time_list = []
	for i in range (len(list_pos_i)):
		pos_i = [list_pos_i[i][0],list_pos_i[i][1],list_pos_i[i][2]]
		pos_f = [list_pos_f[i][0],list_pos_f[i][1],list_pos_f[i][2]]
		start_time = time.time()
		if graph_disp:
			plt.subplot(2,5,count)
			title = "Start "+str(count)
			solveDdp(pos_i, pos_f, graph_disp)
			plt.axis([-4.5,1,-4,1.5])	
			plt.ylabel("y (m)")
			plt.xlabel("x (m)")	
			if (i+1)%4 == 0:
				count += 1
		else:
			solveDdp(pos_i, pos_f, graph_disp)
		time_list.append(time.time() - start_time)
	if graph_disp:
		plt.show()
	print(time_list,len(time_list),np.mean(time_list))
	print("End of Compute them all")

def normalizeAngle(angle): 
	new_angle = angle
	while new_angle > pi:
		new_angle -= 2*pi
	while new_angle < -pi:
		new_angle += 2*pi
	return new_angle	

def distanceBetweenCurvs(path_human,path_ddp):
	print("------- Compute Distance -------")
	global_dist = 0
	global_dist_ang = 0
	final_dist = 0

	for i in range (len(path_human)):
		# print(i,path_human_list[i],path_ddp_list[i])
		human_data = np.loadtxt(path_human_list[i])
		# x_real, y_real, th_real = human_data[0],human_data[1],human_data[5] # th_global
		x_real, y_real, th_real = human_data[0],human_data[1],human_data[4] # th_local		
		ddp_data = np.transpose(np.loadtxt(path_ddp_list[i]))
		x_sim, y_sim, th_sim = ddp_data[0], ddp_data[1], ddp_data[2]
		distance_lin = 0
		distance_ang = 0
		okay = np.where(np.abs(np.diff(x_sim)) + np.abs(np.diff(y_sim)) > 0)

		x_sim = x_sim[okay]
		y_sim = y_sim[okay]
		tck, u = splprep([x_sim, y_sim], s=0)
		length = len(x_real)
		# print(length)
		unew = np.linspace(0,1,length)
		data_sim = splev(unew, tck)

		okay = np.where(np.abs(np.diff(x_real)) + np.abs(np.diff(y_real)) > 0)
		x_real = x_real[okay]
		y_real = y_real[okay]
		tck, u = splprep([x_real, y_real], s=0)
		unew = np.linspace(0,1,length)
		data_real = splev(unew, tck)		
		x_real,y_real = data_real[0],data_real[1]

		for i in range (length):
			distance_lin += np.sqrt((data_sim[0][i]-x_real[i])**2+(data_sim[1][i]-y_real[i])**2)	
			# print(distance_lin)
		# 	if i%25 == 0:
		# 		print(np.sqrt((data_sim[0][i]-x_real[i])**2+(data_sim[1][i]-y_real[i])**2))
		# 		plt.plot([data_sim[0][i],x_real[i]], [data_sim[1][i],y_real[i]], color = 'black', linewidth = 0.5) 	
		# plt.plot(x_sim,y_sim,color='blue')
		# plt.plot(x_real,y_real,color='red')
		# plt.plot(data_sim[0], data_sim[1],color='cyan',linestyle=':', marker='o')
		# # plt.plot(data_real[0], data_real[1],color='orange',linestyle=':', marker='o')	
		# plt.show()

		th_sim = np.interp(np.arange(0,length,1),np.linspace(0,length,len(th_sim)),th_sim)
		
		# 	delta_y = np.diff(y)
		# delta_x = np.diff(x)	
		# th_local = []
		# for i in range (len(delta_x)):
		# phi = atan2(delta_y[i],delta_x[i])

		# th_local.append(phi-theta[i])

		for i in range (len(th_real)-1):
			phi = atan2(data_sim[1][i+1]-data_sim[1][i],data_sim[0][i+1]-data_sim[0][i])
			th_local = phi - th_sim[i]
			# distance_ang += abs(normalizeAngle(th_real[i])-normalizeAngle(th_sim[i]))
			distance_ang += abs(th_real[i]-th_local)

		# print("dist_i :",distance_lin/length,len(okay[0]),distance_ang/len(th_real))
		global_dist += (distance_lin/length)
		global_dist_ang += (distance_ang/(len(th_real)-1)/2)

	print ("final distance :",(global_dist)/len(path_human),(global_dist_ang)/len(path_human))
	print("------- End of Compute distance -------")
	return (global_dist+global_dist_ang)/len(path_human)

def ioc(wt):
	wt =[np.float32(w) for w in wt]
	if (wt[0] >= 0 and wt[1] >= 0 and wt[2] >= 0 and wt[3] >= 0 and\
	 wt[4] >= 0 and wt[5] >= 0 and wt[6] >= 0 and wt[7] >= 0 and wt[8] >= 0):
		model.costWeights = np.matrix(wt[:5]).T
		terminal_model.costWeights = np.matrix(wt[5:]).T
		print("Weights :", model.costWeights,terminal_model.costWeights)
		computeMultipleTraj(init_pos_list, final_pos_list, graph_disp)
		return distanceBetweenCurvs(path_human_list, path_ddp_list)
	else:
		return 100

########################################################################
################################## MAIN ################################
########################################################################

graph_disp = True

model = crocoddyl.ActionRunningModelHuman()
data  = model.createData()
model.alpha = 1

terminal_model = crocoddyl.ActionTerminalModelHuman()
terminal_data  = terminal_model.createData()
terminal_model.alpha = 1

init_pos_list = []
orientation = [pi/2,0,-pi/2,pi]
pos_f = [0,0,pi/2]
final_pos_list = []
start_and_end = np.loadtxt("data/Human/StartAndEnd.dat")
for i in range(len(start_and_end)):
	init_pos_list.append([floor(start_and_end[i][0]*1000)/1000,floor(start_and_end[i][1]*1000)/1000,orientation[i%4]])
	final_pos_list.append([floor(start_and_end[i][2]*1000)/1000,floor(start_and_end[i][3]*1000)/1000,pi/2])

## Easy Compute of Optimal Path ########################################

# model.costWeights = np.matrix([1.,1.2,1.7,.7,5.2,5,8]).T
# i = 36
# solveDdp([init_pos_list[i][0],init_pos_list[i][1],orientation[i%4]], pos_f)
# plt.show()
# computeMultipleTraj(init_pos_list, pos_f, graph_disp)

########################################################################

direction_list = ['N','E','S','O']
position_list = ['1500','4000','-0615','0615','1515','4015','-0640','0640','1540','4040']
path_human_list = []
for pos in position_list:
	for direction in direction_list:
		name_file = direction+pos+".dat"
		path_human_list.append('data/Human/'+name_file)

fin_pos = [0,0,1.57]
orientation_list = [1.57,0.0,-1.58,3.14]
path_ddp_list = []
i = 0
for pos in init_pos_list:
	name_file = 'DdpResult_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	'_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_pos.dat'
	path_ddp_list.append('data/DdpResult/'+name_file)
	i += 1

# path_human_list = path_human_list[28:36]
# print(path_human_list)
# path_ddp_list = path_ddp_list[28:36]
# init_pos_list = init_pos_list[28:36]
# final_pos_list = init_pos_list[28:36]

# ioc([  2.64997930e+00,   3.99903860e+00,   2.00001029e+01,\
#          2.89987185e-07,   1.00000000e+01,   1.00000025e+01,\
#          9.99999967e+00,   3.81925153e-01,   1.37999995e+00]) # results for d_xy
# ioc([  7.87,   4.13,   20.146,\
#          1e-06,   10.,   9.99,\
#          9.99,   3.80e-01,   3.36]) # results for d_xy + 1/2 d_theta
ioc([  7.86951486e+00,   4.00027971e+00,   2.01459991e+01,\
         1.00000000e-06,   9.99999967e+00,   9.98999939e+00,\
         9.98999934e+00,   3.79999984e-01,   3.35999389e+00])

# distanceBetweenCurvs(path_human_list, path_ddp_list)

# wt0 = [  7.87,   4.,   20.146,\
#          1e-06,   10.,   9.99,\
#          9.99,   3.80e-01,   3.36]
# optimal_wt = minimize(ioc, wt0, method='Powell')
# print(optimal_wt)

#########################################################################
######################## OLD FUNCTION DEFINITION  #######################
#########################################################################

# def distanceBetweenCurvsSpeed(path_human,path_ddp):
# 	print("------- Compute Distance -------")
# 	global_dist = 0
# 	final_dist = 0
	
# 	for i in range (len(path_human)):
# 		human_data = np.loadtxt(path_human_list[i])

# 		v = np.sqrt(human_data[2]**2+human_data[3]**2)
# 		ind_begin = np.where(v[:2*len(v)/10] > 0.1)
# 		ind_end= np.where(v[8*len(v)/10:] < 0.1)
# 		if len(ind_begin[0]) != 0 and len(ind_end[0]) != 0:
# 			begin = ind_begin[0][0]
# 			end = 8*len(v)/10+ind_end[0][0]
# 		elif len(ind_begin[0]) == 0:
# 			begin = 2*len(v)/10
# 			if len(ind_end[0]) == 0:
# 				end = len(v)-1
# 			else :
# 				end = 8*len(v)/10+ind_end[0][0]
# 		else:
# 			end = len(v)-1
# 			begin = ind_begin[0][0]
# 		x_real, y_real, th_real = human_data[0][begin:end],human_data[1][begin:end],human_data[5][begin:end]
# 		# time = np.arange(0, len(v), 1)
# 		# plt.plot(time,v)
# 		v_real = v[begin:end]
# 		# plt.plot(time[begin:end],v_real)
# 		# plt.show()

# 		length = len(x_real)
# 		# print(length)

# 		ddp_data = np.transpose(np.loadtxt(path_ddp_list[i]))
# 		x_sim, y_sim, th_sim = ddp_data[0], ddp_data[1], ddp_data[2]

		
# 		distance_lin = 0
# 		distance_ang = 0
# 		distance_speed = 0
# 		okay = np.where(np.abs(np.diff(x_sim)) + np.abs(np.diff(y_sim)) > 0)
# 		x_sim = x_sim[okay]
# 		y_sim = y_sim[okay]
# 		tck, u = splprep([x_sim, y_sim], s=0)
# 		unew = np.linspace(0,1,length)
# 		data_sim = splev(unew, tck)

# 		v_sim = np.sqrt((np.diff(x_sim)/0.1)**2+(np.diff(y_sim)/0.1)**2)
# 		okay = np.where(np.abs(np.diff(v_sim)) > 0)
# 		v_sim = v_sim[okay]
# 		tck, u = splprep([v_sim], s=0)
# 		unew = np.linspace(0,1,length)
# 		v_sim = splev(unew, tck)		

# 		# time = np.arange(0, length, 1)
# 		# plt.plot(time,v_real,color='cyan')
# 		# plt.plot(time,v_sim[0],color='orange')
# 		# plt.show()


# 		for i in range (length):
# 			distance_lin += np.sqrt((data_sim[0][i]-x_real[i])**2+(data_sim[1][i]-y_real[i])**2)
# 			distance_speed += np.sqrt((v_sim[0][i]-v_real[i])**2)

# 			# if i%25 == 0:
# 			# 	plt.plot([data_sim[0][i],x_real[i]], [data_sim[1][i],y_real[i]], color = 'black', linewidth = 0.5) 	
# 		# plt.plot(x_sim,y_sim,color='blue')
# 		# plt.plot(x_real,y_real,color='red')
# 		# # plt.plot(data_sim[0], data_sim[1],color='cyan',linestyle=':', marker='o')
# 		# # plt.plot(data_real[0], data_real[1],color='orange',linestyle=':', marker='o')	
# 		# plt.show()

# 		okay = np.where(np.abs(np.diff(th_sim)) > 0)
# 		th_sim = th_sim[okay]		
# 		tck, u = splprep([th_sim], s=0)
# 		data = splev(unew, tck)	
# 		th_sim = data[0]
		
# 		for i in range (len(th_real)):
# 			distance_ang += abs(normalizeAngle(th_real[i])-normalizeAngle(th_sim[i]))
		
# 		print("dist_i :",distance_lin/length,distance_ang/length,distance_speed/length,len(okay[0]))#,distance_ang/len(th_real))
# 		global_dist += (distance_lin)/length
# 		# final_dist += np.sqrt((data_sim[0][-1]-x_real[-1])**2 +(data_sim[1][-1]-y_real[-1])**2)
# 		# global_dist += (distance_ang/length)
# 		# final_dist += abs(normalizeAngle(th_real[-1])-normalizeAngle(th_sim[-1]))
# 		# else:
# 		# 	final_dist += 10
# 		# 	print("not okay")
# 	print ("final distance :",(global_dist)/len(path_human))
# 	print("------- End of Compute distance -------")
# 	return (global_dist)/len(path_human)