import numpy as np
import matplotlib.pylab as plt
from math import pi,floor
from scipy.interpolate import splprep, splev

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
#########################################################################################
################################## FCT DEFINITION #######################################
#########################################################################################

def readModel(path,time):
	curv = np.transpose(np.loadtxt(path))
	x = curv[0]
	y = curv[1]
	theta = curv[2]
	#theta = curv[2]
	time_model = np.linspace(0,100,len(x))
	x = np.interp(time,time_model,x)
	y = np.interp(time,time_model,y)
	return (x,y,theta)

def distanceBetweenCurvs(x_real,x_sim,y_real,y_sim):
	infx_real,infx_sim,supx_real,supx_sim = min(x_real),min(x_sim),max(x_real),max(x_sim)
	infy_real,infy_sim,supy_real,supy_sim = min(y_real),min(y_sim),max(y_real),max(y_sim)	
	distance = 0
	length = 100

	tck, u = splprep([x_sim, y_sim], s=0)
	unew = np.linspace(0,1,length)
	data_sim = splev(unew, tck)
	tck, u = splprep([x_real, y_real], s=0)
	data_real = splev(unew, tck)

	for i in range (length):
		distance += np.sqrt((data_sim[0][i]-data_real[0][i])**2+(data_sim[1][i]-data_real[1][i])**2)
		# print(distance)	 	
	# plt.plot(x_sim,y_sim,color='blue')
	# plt.plot(x_real,y_real,color='green')
	# plt.plot(data_sim[0], data_sim[1],color='red',linestyle=':', marker='o')
	# plt.plot(data_real[0], data_real[1],color='orange',linestyle=':', marker='o')	
	# plt.show()				
	return distance/length	


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
	name_file = 'Clothoid_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	'_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_0.001_pos.dat'
	path_clothoid_list.append('data/Clothoid/'+name_file)
	name_file = 'DdpResult_from_'+str(pos[0])+','+str(pos[1])+','+str(orientation_list[i%4])+\
	'_to_'+str(fin_pos[0])+','+str(fin_pos[1])+','+str(fin_pos[2])+'_pos.dat'
	path_ddp_list.append('data/DdpResult/'+name_file)
	i += 1

time = np.arange(0,100,1)

# for i in range (0,20):
# 	# i =2
# 	print(i)
# 	human_data = np.loadtxt(path_human_list[i])
# 	(x_clothoid,y_clothoid,theta_clothoid) = readModel(path_ddp_list[i],time)
# 	print("---> ",distanceBetweenCurvs(human_data[0],x_clothoid,human_data[1],y_clothoid))

count = 1
dist_clothoid_list, dist_ddp_list = [],[]
# i = 7
for i in range (len(path_human_list)):
	#print(path_human_list[i],path_clothoid_list[i])
	title = path_human_list[i][11:17]
	#print(title)
	#plt.subplot(1,4,count)
	plt.subplot(4,10,count)
	#if title == 'E1515.' or title == 'N4040.' or title == 'O4015.' or title == 'S-0615':
	print(i,count)		
	human_data = np.loadtxt(path_human_list[i])
	(x_clothoid,y_clothoid,theta_clothoid) = readModel(path_clothoid_list[i],time)
	(x_ddp,y_ddp,theta_ddp) = readModel(path_ddp_list[i],time)	

	plt.plot(x_clothoid,y_clothoid,label='Clothoid',color='red')
	plt.arrow(x_clothoid[-1], y_clothoid[-1], np.cos(theta_clothoid[-1]) * 0.1, np.sin(theta_clothoid[-1]) * 0.1, head_width=.05)
	plt.arrow(x_clothoid[0], y_clothoid[0], np.cos(theta_clothoid[0]) * 0.1, np.sin(theta_clothoid[0]) * 0.1, head_width=.05)

	plt.plot(x_ddp,y_ddp,label='Ddp',color='blue')	

	
	plt.plot(human_data[2],human_data[3],label='Subjects',color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[4],human_data[5],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[6],human_data[7],color='lime',linewidth=0.75,alpha = 0.4)	
	plt.plot(human_data[8],human_data[9],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[10],human_data[11],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[12],human_data[13],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[14],human_data[15],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[16],human_data[17],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[18],human_data[19],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[20],human_data[21],color='lime',linewidth=0.75,alpha = 0.4)
	plt.plot(human_data[0],human_data[1],label='Human average',color='green')
	
	# plt.title(path_human_list[i][11:17])
	dist_clotho = distanceBetweenCurvs(human_data[0],x_clothoid,human_data[1],y_clothoid)
	print("---------------")
	dist_ddp = distanceBetweenCurvs(human_data[0],x_ddp,human_data[1],y_ddp)
	dist_clothoid_list.append(dist_clotho)
	dist_ddp_list.append(dist_ddp)
	#print(i,path_human_list[i][62:68],dist_clotho,dist_ddp)
	#plt.legend()	
	#plt.title('Clothoid-Human :'+str(floor(dist_clotho*100)/100) + \
	#', OC-Human :'+str(floor(dist_ddp*100)/100))
	plt.title('clotho :'+str(floor(dist_clotho*1000)/1000) + \
	' VS ddp :'+str(floor(dist_ddp*1000)/1000))		
	#plt.ylabel("y (m)")
	#plt.xlabel("x (m)")	
	# if count < 4:
	count += 1
plt.show()
print(dist_clothoid_list)
print(dist_ddp_list)

path = "data/dist_clotho.dat"
np.savetxt(path,dist_clothoid_list)

path = "data/dist_ddp.dat"
np.savetxt(path,dist_ddp_list)
