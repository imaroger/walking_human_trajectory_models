import numpy as np
import matplotlib.pylab as plt
from math import pi,floor,atan2,atan
from scipy.interpolate import splprep, splev

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#########################################################################################
################################## FCT DEFINITION #######################################
#########################################################################################

def readModel(path,time):
	length = 500
	curv = np.transpose(np.loadtxt(path))
	x = curv[0]
	y = curv[1]
	theta = curv[2]
	time_model = np.linspace(0,length,len(x))

	# v_norm = np.sqrt((np.diff(x)/0.1)**2+(np.diff(y)/0.1)**2)
	# print(v_norm[-1])

	okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
	x = x[okay]
	y = y[okay]
	tck, u = splprep([x, y], s=0)
	unew = np.linspace(0,1,length)
	data = splev(unew, tck)
	x,y = data[0],data[1]

	theta = np.interp(time,time_model,theta)

	return (x,y,theta)

def distanceBetweenCurvs(x_real,x_sim,y_real,y_sim):
	distance = 0
	length = len(x_real)

	distance_fin = np.sqrt((x_sim[-1]-x_real[-1])**2+(y_sim[-1]-y_real[-1])**2)

	for i in range (length):
		distance += np.sqrt((x_sim[i]-x_real[i])**2+(y_sim[i]-y_real[i])**2)
	# 	if i%25 == 0
	# 		print(i, "sim",x_sim[i],y_sim[i])
	# 		print(np.sqrt((x_sim[i]-x_real[i])**2+(y_sim[i]-y_real[i])**2))
	# 		plt.plot([x_sim[i],x_real[i]], [y_sim[i],y_real[i]], color = 'black', linewidth = 0.5)
	# # 	# print(distance)	 	
	# plt.plot(x_sim,y_sim,color='blue')
	# plt.plot(x_real,y_real,color='red')
	# plt.plot(x_sim, y_sim,color='cyan',linestyle=':', marker='o')
	# plt.plot(x_real, y_real,color='orange',linestyle=':', marker='o')	
	# plt.show()		
	# print("dist_i :",distance/500)		
	return distance/length,distance_fin	

def normalizeAngle(angle): 
	new_angle = angle
	while new_angle > pi:
		new_angle -= 2*pi
	while new_angle < -pi:
		new_angle += 2*pi
	return new_angle	

def angularDistanceBetweenCurvs(th_real,th_sim):
	distance = 0	

	distance_fin = abs(pi/2-normalizeAngle(th_sim[-1]))

	for i in range (len(th_real)):
		distance += abs(normalizeAngle(th_real[i])-normalizeAngle(th_sim[i]))
		# if i%50 == 0:
		# 	print(distance,th_real[i],th_sim[i])
	return distance/len(th_real),distance_fin

#########################################################################################
################################## MAIN #################################################
#########################################################################################

direction_list = ['N','E','S','O']
position_list = ['1500','4000','-0615','0615','1515','4015','-0640','0640','1540','4040']
path_human_list = []
for pos in position_list:
	for direction in direction_list:
		name_file = direction+pos+".dat"
		path_human_list.append('data/Human/'+name_file)

init_pos_list = []
start_and_end = np.loadtxt("data/Human/StartAndEnd.dat")
for i in range(len(start_and_end)):
	init_pos_list.append([floor(start_and_end[i][0]*1000)/1000,floor(start_and_end[i][1]*1000)/1000])
fin_pos = [0,0,1.57]
orientation_list = [1.57,0.0,-1.58,3.14]
path_clothoid_list,path_ddp_list = [],[]
i = 0
for pos in init_pos_list:
	# name_file = 'Clothoid_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	# '_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_0.001_pos.dat'
	# path_clothoid_list.append('data/Clothoid/'+name_file)
	name_file = 'DdpResult_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	'_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_pos.dat'
	path_ddp_list.append('data/DdpResult/'+name_file)
	i += 1

init_pos_list = []
start_and_end = np.loadtxt("data/Human/DataIROS/StartAndEnd.dat")
for i in range(len(start_and_end)):
	init_pos_list.append([floor(start_and_end[i][0]*1000)/1000,floor(start_and_end[i][1]*1000)/1000])
fin_pos = [0,0,1.57]
orientation_list = [1.57,0.0,-1.58,3.14]
path_clothoid_list= []
i = 0
for pos in init_pos_list:
	name_file = 'DdpResult_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	'_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_pos.dat'
	path_clothoid_list.append('data/DdpResult/DataIROS/'+name_file)
	i += 1	

time = np.arange(0,500,1)

fig = plt.figure()
count = 1
dist_clothoid_list, dist_ddp_list,angular_dist_clothoid_list, angular_dist_ddp_list = [],[],[],[]
dist_fin_ddp_list , angular_dist_fin_ddp_list = [],[]
dist_subjects_ddp_list , angular_dist_subjects_ddp_list = [],[]

for i in range (len(path_human_list)):	
	title = path_human_list[i][11:17]
	#print(title)
	# ax = plt.subplot(1,4,count)

	ax = plt.subplot(4,10,count)

	# if title == 'E1540.' or title == 'N-0615' or title == 'S4015.' or title == 'O0640.':
	
	print(title,i,count)		

	human_data = np.loadtxt(path_human_list[i])
	# (x_clothoid,y_clothoid,theta_clothoid) = readModel(path_clothoid_list[i],time)
	(x_ddp,y_ddp,theta_ddp) = readModel(path_ddp_list[i],time)	

	# v = np.sqrt(human_data[2]**2+human_data[3]**2)
	# ind_begin = np.where(v[:2*len(v)/10] > 0.2)
	# ind_end= np.where(v[8*len(v)/10:] < 0.2)
	# if len(ind_begin[0]) != 0 and len(ind_end[0]) != 0:
	# 	begin = ind_begin[0][0]
	# 	end = 8*len(v)/10+ind_end[0][0]
	# elif len(ind_begin[0]) == 0:
	# 	begin = 2*len(v)/10
	# 	if len(ind_end[0]) == 0:
	# 		end = len(v)-1
	# 	else :
	# 		end = 8*len(v)/10+ind_end[0][0]
	# else:
	# 	end = len(v)-1
	# 	begin = ind_begin[0][0]

	# if np.sum(human_data[5]) != 0:
	# 	theta_trunc = human_data[5][begin:end]
	# 	okay = np.where(np.abs(np.diff(theta_trunc)) > 0)
	# 	theta_trunc = theta_trunc[okay]		
	# 	print(len(theta_trunc))
	# 	tck, u = splprep([theta_trunc], s=0)
	# 	unew = np.linspace(0,1,len(human_data[5]))
	# 	data = splev(unew, tck)	
	# 	human_data[5] = data[0]
	# human_data[5] = np.interp(time,np.linspace(0,100,len(theta_trunc)),theta_trunc)

	# plt.plot(x_clothoid,y_clothoid,label='Clothoid',color='red',linewidth=1.5)
	# # plt.arrow(x_clothoid[-1], y_clothoid[-1], np.cos(theta_clothoid[-1]) * 0.1, np.sin(theta_clothoid[-1]) * 0.1, head_width=.05)
	# # plt.arrow(x_clothoid[0], y_clothoid[0], np.cos(theta_clothoid[0]) * 0.1, np.sin(theta_clothoid[0]) * 0.1, head_width=.05)

	plt.plot(x_ddp,y_ddp,label='OC',color='red',linewidth=1.5)	
	
	plt.plot(human_data[6],human_data[7],label='Subjects',color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[12],human_data[13],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[18],human_data[19],color='lime',linewidth=0.75,alpha = 0.4)	
	plt.plot(human_data[24],human_data[25],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[30],human_data[31],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[36],human_data[37],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[42],human_data[43],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[48],human_data[49],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[54],human_data[55],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[60],human_data[61],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[0],human_data[1],label='Human average',color='green',linewidth=1.5)

	if np.sum(human_data[5]) != 0:
		arrow_len = 0.2
		for i in range (len(human_data[0])):
			if i%50 == 0:
				plt.arrow(human_data[0][i], human_data[1][i], np.cos(human_data[5][i])*arrow_len, np.sin(human_data[5][i])*arrow_len, head_width=.03,color='green')
				plt.arrow(x_ddp[i], y_ddp[i], np.cos(theta_ddp[i])*arrow_len, np.sin(theta_ddp[i])*arrow_len, head_width=.03,color='red')
	plt.arrow(x_ddp[-1], y_ddp[-1], np.cos(theta_ddp[-1])*arrow_len, np.sin(theta_ddp[-1])*arrow_len, head_width=.03,color='red')
	# plt.plot(time,v,color='orange')
	# plt.plot([time[end]]*len(time),np.linspace(0,6,len(time)),color ='black')
	# plt.plot([time[begin]]*len(time),np.linspace(0,6,len(time)),color ='black')	
	# plt.plot(time,human_data[5],linestyle=':',color ='black')
	# plt.plot(time,theta_clothoid,color='red')
	# plt.plot(time,theta_ddp,color='blue')	
	# plt.plot(time,theta_trunc,color='green')
	# plt.plot(time,human_data[11],color='lime',linewidth=0.75,alpha = 0.4)
	# plt.plot(time,human_data[17],color='lime',linewidth=0.75,alpha = 0.4)	
	# plt.plot(time,human_data[23],color='lime',linewidth=0.75,alpha = 0.4)
	# plt.plot(time,human_data[29],color='lime',linewidth=0.75,alpha = 0.4)
	# plt.plot(time,human_data[35],color='lime',linewidth=0.75,alpha = 0.4)	
	# plt.plot(time,human_data[41],color='lime',linewidth=0.75,alpha = 0.4)
	# plt.plot(time,human_data[47],color='lime',linewidth=0.75,alpha = 0.4)	
	# plt.plot(time,human_data[53],color='lime',linewidth=0.75,alpha = 0.4)
	# plt.plot(time,human_data[59],color='lime',linewidth=0.75,alpha = 0.4)	
	# plt.plot(time,human_data[65],color='lime',linewidth=0.75,alpha = 0.4)

	# dist_clotho = distanceBetweenCurvs(human_data[0],x_clothoid,human_data[1],y_clothoid)
	dist_ddp,dist_fin_ddp = distanceBetweenCurvs(human_data[0],x_ddp,human_data[1],y_ddp)

	# dist_clothoid_list.append(dist_clotho)
	dist_ddp_list.append(dist_ddp)
	dist_fin_ddp_list.append(dist_fin_ddp)

	for i in range (10):
		dist_subjects_ddp_list.append(distanceBetweenCurvs(human_data[6+i*6],x_ddp,human_data[7+i*6],y_ddp)[0])
	
	if np.sum(human_data[5]) != 0:
		print("yes")
		# angular_dist_clotho = angularDistanceBetweenCurvs(human_data[5],theta_clothoid)
		angular_dist_ddp,angular_dist_fin_ddp = angularDistanceBetweenCurvs(human_data[5],theta_ddp)
	else:
		# angular_dist_clotho = 0
		angular_dist_ddp,angular_dist_fin_ddp = 0,0				
		print("no",title)

	# angular_dist_clothoid_list.append(angular_dist_clotho)
	angular_dist_ddp_list.append(angular_dist_ddp)
	angular_dist_fin_ddp_list.append(angular_dist_fin_ddp)

	#print(i,path_human_list[i][62:68],dist_clotho,dist_ddp)
	# plt.legend(fontsize = 'xx-large')	
	plt.title(title)
	# plt.title("d_xy = " + str(floor(dist_ddp*10000)/10000) + " & d_theta = "+str(floor(angular_dist_ddp*10000)/10000))

	# plt.title('clotho :'+str(floor(angular_dist_clotho*100)/100) + \
	# ' VS ddp :'+str(floor(angular_dist_ddp*100)/100))
	# plt.title('Clothoid-Human d_xy='+str(floor(dist_clotho*100)/100) + \
	# ' & d_th='+str(floor(angular_dist_clotho*100)/100)+ \
	# ', OC-Human d_xy='+str(floor(dist_ddp*100)/100)+ \
	# ' & d_th='+str(floor(angular_dist_ddp*100)/100))	
	# ax.set_xticklabels([])
	# ax.set_yticklabels([])
	plt.ylabel("y (m)")
	plt.xlabel("x (m)")	
	# if count < 4:
	count += 1
plt.show()

# path = "data/dist_clotho.dat"
# np.savetxt(path,dist_clothoid_list)

path = "data/dist_ddp.dat"
np.savetxt(path,dist_ddp_list)

path = "data/dist_fin_ddp.dat"
np.savetxt(path,dist_fin_ddp_list)

# path = "data/angular_dist_clotho.dat"
# np.savetxt(path,angular_dist_clothoid_list)

path = "data/angular_dist_ddp.dat"
np.savetxt(path,angular_dist_ddp_list)

path = "data/angular_dist_fin_ddp.dat"
np.savetxt(path,angular_dist_fin_ddp_list)

path = "data/dist_subjects_ddp.dat"
np.savetxt(path,dist_subjects_ddp_list)