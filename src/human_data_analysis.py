import numpy as np
import matplotlib.pylab as plt
from math import pi,floor,atan2,atan
from scipy.signal import butter, filtfilt
from scipy.interpolate import splprep, splev

#########################################################################################
################################## FCT DEFINITION #######################################
#########################################################################################

def continuousAtan(y,x):
	l_begin = [atan2(y[len(y)/2],x[len(y)/2])]
	last = l_begin[-1]
	
	i = len(y)/2-1
	while i  >= 0:
		j = 0
		while abs(atan2(y[i],x[i])-last) > pi/2 and i > 0:
			i -= 1
			j += 1
		if j !=0:
			for k in range(j):
				new_atan = atan2(y[i+j-k],x[i+j-k])
				if last > 0:
					new_atan = new_atan + 2*pi
				else :
					new_atan = new_atan - 2* pi
				l_begin.append(new_atan)
		else:			
			l_begin.append(atan2(y[i],x[i]))						
			last = l_begin[-1]
			i -= 1
	i = len(y)/2+2
	l_end = [atan2(y[len(y)/2+1],x[len(y)/2+1])]
	last = l_end[-1]
	while i <= len(y)-1:
		j = 0
		while abs(atan2(y[i],x[i])-last) > pi/2 and i < len(y)-1:
			i += 1
			j += 1
		if j != 0:
			for k in range(j):
				new_atan = atan2(y[i-j+k],x[i-j+k])
				if last > 0:
					new_atan = new_atan + 2*pi
				else :
					new_atan = new_atan - 2* pi
				l_end.append(new_atan)
		else:
			l_end.append(atan2(y[i],x[i]))
			last = l_end[-1]
			i += 1
	return np.concatenate((np.flip(l_begin,0), l_end))

def distanceBetweenCurvs(x_real,x_mean,y_real,y_mean):
	distance = 0
	length = 500

	okay = np.where(np.abs(np.diff(x_real)) + np.abs(np.diff(y_real)) > 0)
	x_real = x_real[okay]
	y_real = y_real[okay]
	tck, u = splprep([x_real, y_real], s=0)
	unew = np.linspace(0,1,length)
	data = splev(unew, tck)
	x_real,y_real = data[0],data[1]

	okay = np.where(np.abs(np.diff(x_mean)) + np.abs(np.diff(y_mean)) > 0)
	x_mean = x_mean[okay]
	y_mean = y_mean[okay]
	tck, u = splprep([x_mean, y_mean], s=0)
	data = splev(unew, tck)
	x_mean,y_mean = data[0],data[1]

	distance_fin = np.sqrt((x_mean[-1]-x_real[-1])**2+(y_mean[-1]-y_real[-1])**2)

	for i in range (length):
		distance += np.sqrt((x_mean[i]-x_real[i])**2+(y_mean[i]-y_real[i])**2)
		
	# 	if i%25 == 0:
	# 		print(i, "sim",x_mean[i],y_mean[i])
	# 		print(np.sqrt((x_mean[i]-x_real[i])**2+(y_mean[i]-y_real[i])**2))
	# 		plt.plot([x_mean[i],x_real[i]], [y_mean[i],y_real[i]], color = 'black', linewidth = 0.5)
	# # 	# print(distance)	 	
	# plt.plot(x_mean,y_mean,color='blue')
	# plt.plot(x_real,y_real,color='red')
	# plt.plot(x_mean, y_mean,color='cyan',linestyle=':', marker='o')
	# plt.plot(x_real, y_real,color='orange',linestyle=':', marker='o')	
	# plt.show()		
	# print("dist_i :",distance/500)		
	return distance/length

def angularDistanceBetweenCurvs(th_real,th_sim):
	distance = 0		
	for i in range (len(th_real)):
		distance += abs(normalizeAngle(th_real[i])-normalizeAngle(th_sim[i]))
	return distance/len(th_real)


# def readHuman(path,time):
# 	curv = np.transpose(np.loadtxt(path))
# 	x = curv[1]*0.001-3.23824
# 	y = curv[2]*0.001-0.69981
# 	vx = curv[4]*0.001
# 	vy = curv[5]*0.001
# 	time_human = np.linspace(0,100,len(x))
# 	x = np.interp(time,time_human,x)
# 	y = np.interp(time,time_human,y)
# 	vx = np.interp(time,time_human,vx)
# 	vy = np.interp(time,time_human,vy)	
# 	# plt.plot(time,vx, linestyle=':',color='black')
# 	b, a = butter(5, 0.03, btype='lowpass')
# 	# vx = filtfilt(b, a, vx)
# 	# plt.plot(time,vx)	
# 	# plt.show()

# 	# plt.plot(time,vy, linestyle=':',color='black')
# 	b, a = butter(5, 0.03, btype='lowpass')
# 	# vy = filtfilt(b, a, vy)
# 	# plt.plot(time,vy)	
# 	# plt.show()
# 	if np.isnan(curv[-1][0]) == False and len(curv) == 9:
# 		theta = curv[-1]*pi/180
# 		th_local = np.interp(time,time_human,theta)
# 		# ind_end = int(len(v)/2)
# 		# # print(ind,y[ind+1]-y[ind],x[ind+1]-x[ind])
# 		# while ind_end < len(v) and abs(v[ind_end]) > 0.3:
# 		# 	ind_end += 1
# 		# print(ind_end)
# 		ind_end = len(time)-1		

# 		l = continuousAtan(np.diff(y),np.diff(x))
# 		tangent = l#np.arctan(np.diff(y)/np.diff(x))
# 		# plt.plot(time[:-1],np.arctan2(np.diff(y),np.diff(x)), linestyle=':',color='green')
		
# 		# # plt.plot(time[:-1],np.arctan(np.diff(y)/np.diff(x)), linestyle=':',color='red')
# 		# plt.plot(time[:-1],tangent, linestyle=':',color='black')			
# 		b, a = butter(5, 0.038, btype='lowpass')
# 		tangent = filtfilt(b, a, tangent)
		
# 		th_global = [tangent[i]+th_local[i] for i in range (ind_end)]
# 		th_global += [th_global[-1]]		
# 		# plt.plot(time,th_local, linestyle=':',color='red')		
# 		# plt.plot(time[:-1],tangent)		
# 		# plt.plot(time,th_global)
# 		# # plt.plot(time,v)
# 		# plt.show()
# 		print(path,"yes")

# 	else:
# 		th_local = np.zeros(len(x))
# 		th_global = np.zeros(len(x))
# 		print(path,"no") 
# 	return (x,y,vx,vy,th_local,th_global)

def readHuman(path,time):
	curv = np.transpose(np.loadtxt(path))
	x = curv[1]*0.001-3.23824
	y = curv[2]*0.001-0.69981
	vx = curv[4]*0.001
	vy = curv[5]*0.001
	time_human = np.linspace(0,end,len(x))

	# vx = np.interp(time,time_human,vx)
	# vy = np.interp(time,time_human,vy)

	# plt.subplot(2,3,1)
	# plt.plot(time_human,x)
	# plt.plot(time_human,y)
	# plt.plot(time_human,vx)
	# plt.plot(time_human,vy)	

	# plt.subplot(2,3,4)
	# plt.plot(x,y)

	# okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
	# x = x[okay]
	# y = y[okay]		
	# tck, u = splprep([x, y], s=0)
	length = len(time)
	# unew = np.linspace(0,1,length)
	# data = splev(unew, tck)	
	# x,y = data[0],data[1]

	# okay = np.where(np.abs(np.diff(vx)) + np.abs(np.diff(vy)) > 0)
	# vx = vx[okay]
	# vy = vy[okay]		
	# tck, u = splprep([vx, vy], s=0)
	# data = splev(unew, tck)	
	# vx,vy = data[0],data[1]
	
	# plt.subplot(2,3,2)
	# plt.plot(time,x)
	# plt.plot(time,y)
	# plt.plot(time,vx)
	# plt.plot(time,vy)	
	# plt.subplot(2,3,5)
	# plt.plot(x,y)

	# x = curv[1]*0.001-3.23824
	# y = curv[2]*0.001-0.69981
	# vx = curv[4]*0.001
	# vy = curv[5]*0.001
		
	x = np.interp(time,time_human,x)
	y = np.interp(time,time_human,y)
	vx = np.interp(time,time_human,vx)
	vy = np.interp(time,time_human,vy)	
	
	# plt.subplot(2,3,3)
	# plt.plot(time,x)
	# plt.plot(time,y)
	# plt.plot(time,vx)
	# plt.plot(time,vy)	

	# plt.subplot(2,3,6)
	# plt.plot(x,y)
	# plt.show()

	# plt.plot(time,vx, linestyle=':',color='black')
	b, a = butter(5, 0.03, btype='lowpass')
	# vx = filtfilt(b, a, vx)
	# plt.plot(time,vx)	
	# plt.show()

	# plt.plot(time,vy, linestyle=':',color='black')
	b, a = butter(5, 0.03, btype='lowpass')
	# vy = filtfilt(b, a, vy)
	# plt.plot(time,vy)	
	# plt.show()
	if np.isnan(curv[-1][0]) == False and len(curv) == 9:
		theta = curv[-1]*pi/180
		# th_local = np.interp(time,time_human,theta)
		# plt.subplot(1,3,1)
		# plt.plot(time_human,theta)

		# okay = np.where(np.abs(np.diff(theta)) > 0)
		# th_local = theta[okay]		
		# tck, u = splprep([theta], s=0)
		# unew = np.linspace(0,1,length)
		# data = splev(unew, tck)	
		# th_local = data[0]
		
		
		# plt.subplot(1,3,2)
		# plt.plot(time,th_local)	

		# theta = curv[-1]*pi/180
		th_local = np.interp(time,time_human,theta)
		# plt.subplot(1,3,3)
		# plt.plot(time,th_local)	
		# plt.show()		

		# ind_end = int(len(v)/2)
		# # print(ind,y[ind+1]-y[ind],x[ind+1]-x[ind])
		# while ind_end < len(v) and abs(v[ind_end]) > 0.3:
		# 	ind_end += 1
		# print(ind_end)
		ind_end = len(time)-1		

		l = continuousAtan(np.diff(y),np.diff(x))
		tangent = l#np.arctan(np.diff(y)/np.diff(x))
		# plt.plot(time[:-1],np.arctan2(np.diff(y),np.diff(x)), linestyle=':',color='green')
		
		# # plt.plot(time[:-1],np.arctan(np.diff(y)/np.diff(x)), linestyle=':',color='red')
		# plt.plot(time[:-1],tangent, linestyle=':',color='black')			
		b, a = butter(5, 0.038, btype='lowpass')
		tangent = filtfilt(b, a, tangent)
		
		th_global = [tangent[i]+th_local[i] for i in range (ind_end)]
		th_global += [th_global[-1]]		
		# plt.plot(time,th_local, linestyle=':',color='red')		
		# plt.plot(time[:-1],tangent)		
		# plt.plot(time,th_global)
		# # plt.plot(time,v)
		# plt.show()
		print(path,"yes")

	else:
		th_local = np.zeros(len(x))
		th_global = np.zeros(len(x))
		print(path,"no") 
	return (x,y,vx,vy,th_local,th_global)

def writeMeanTrajectory(pos_ind,time,nb_subject):
	data = np.zeros(((nb_subject+1)*6,len(time)))
	title = path_human_list[pos_ind][60:66]
	#print(title)
	if title[5] == '_':
		title = title[0:5]
	nb_subject_with_angle = 0
	for i in range (6,(nb_subject+1)*6,6):
		(data[i],data[i+1],data[i+2],data[i+3],data[i+4],data[i+5]) = np.array(readHuman(path_human_list[pos_ind+(i/6-1)*40],time))
		if (np.sum(data[i+4]) != 0):
			nb_subject_with_angle += 1
	print(nb_subject_with_angle)
	data[0] = (data[6]+data[12]+data[18]+data[24]+data[30]+data[36]+data[42]\
	+data[48]+data[54]+data[60])/nb_subject
	data[1] = (data[7]+data[13]+data[19]+data[25]+data[31]+data[37]+data[43]\
	+data[49]+data[55]+data[61])/nb_subject
	data[2] = (data[8]+data[14]+data[20]+data[26]+data[32]+data[38]+data[44]\
	+data[50]+data[56]+data[62])/nb_subject

	# plt.plot(time,data[2], linestyle=':',color='black')
	b, a = butter(5, 0.04, btype='lowpass')
	data[2] = filtfilt(b, a, data[2])
	# plt.plot(time,data[2])
	# plt.show()	

	data[3] = (data[9]+data[15]+data[21]+data[27]+data[33]+data[39]+data[45]\
	+data[51]+data[57]+data[63])/nb_subject	

	# plt.plot(time,data[3], linestyle=':',color='black')
	b, a = butter(5, 0.04, btype='lowpass')
	data[3] = filtfilt(b, a, data[3])
	# plt.plot(time,data[3])
	# plt.show()		

	if nb_subject_with_angle != 0:
		data[4] = (data[10]+data[16]+data[22]+data[28]+data[34]+data[40]+data[46]\
		+data[52]+data[58]+data[64])/nb_subject_with_angle	
		data[5] = (data[11]+data[17]+data[23]+data[29]+data[35]+data[41]+data[47]\
		+data[53]+data[59]+data[65])/nb_subject_with_angle
		
	# plt.plot(time,data[5], linestyle=':',color='black')
	# plt.plot(time,data[11], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[17], linewidth = 0.5,color='lime')		
	# plt.plot(time,data[23], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[29], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[35], linewidth = 0.5,color='lime')		
	# plt.plot(time,data[41], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[47], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[53], linewidth = 0.5,color='lime')		
	# plt.plot(time,data[59], linewidth = 0.5,color='lime')	
	# plt.plot(time,data[65], linewidth = 0.5,color='lime')

	b, a = butter(5, 0.025, btype='lowpass')
	data[5] = filtfilt(b, a, data[5])
	# plt.plot(time,data[5])
	# plt.show()
	path = "data/Human/"+title+".dat"
	#print("---->",path)
	np.savetxt(path,data)

def finalMeanComputing(position_list,direction_list):
	mean_x_f,mean_y_f = 0,0
	for pos in position_list:
		for direction in direction_list:
			path = 'data/Human/'+direction+pos+'.dat'
			data = np.loadtxt(path)
			mean_y_f += data[1][-1]
			mean_x_f += data[0][-1]
	return (mean_x_f/40,mean_y_f/40)

def normalizeAngle(angle): 
	new_angle = angle
	while new_angle > pi:
		new_angle -= 2*pi
	while new_angle < -pi:
		new_angle += 2*pi
	return new_angle	

#########################################################################################
################################## MAIN #################################################
#########################################################################################

nb_subject = 10
direction_list = ['N','E','S','O']
position_list = ['1500','4000','-0615','0615','1515','4015','-0640','0640','1540','4040']
path_human_list = []
for i in range (1,nb_subject+1):
	for pos in position_list:
		for direction in direction_list:
			name_file = 'sujet'+str(i)+"_"+direction+pos+'_1.txt'
			if i == 5 and direction == 'N' and pos == '-0615' or \
			i == 4 and direction == 'E' and pos == '-0615' or \
			i == 1 and direction == 'N' and pos == '1515' or  \
			i == 2 and direction == 'E' and pos == '0640' or \
			i == 6 and direction == 'S' and pos == '0615' or \
			i == 8 and direction == 'O' and pos == '0615':
				name_file = 'sujet'+str(i)+"_"+direction+pos+'_2.txt'
			path_human_list.append('/home/imaroger/Documents/Article/IROS2020/Human_data/'+name_file)

end = 100
time = np.arange(0,end,0.2)

# for i in range (15,len(path_human_list),40):
	
# 	(x,y,vx,vy,th_loc,th) = readHuman(path_human_list[i],time)
# 	plt.subplot(1,3,1)
# 	plt.plot(time,vx)
# 	plt.plot(time,vy)
# 	plt.subplot(1,3,2)
# 	plt.plot(time,th_loc)
# 	plt.plot(time,th)
# 	plt.subplot(1,3,3)
# 	plt.plot(x,y)
# 	arrow_len = 0.1
# 	count = 0
# 	for i in range (len(x)-1):
# 		if count%50 == 0:
# 			c, s = np.cos(th[i]), np.sin(th[i])	
# 			plt.arrow(x[i], y[i], c*arrow_len, s*arrow_len, head_width=.005)
# 		count += 1
# 	plt.show()

# print(path_human_list[33])
# writeMeanTrajectory(33,time,nb_subject)


for i in range (0,40):
	writeMeanTrajectory(i,time,nb_subject)

# print(finalMeanComputing(position_list,direction_list))

# start_and_end = []
# for pos in position_list:
# 	for direction in direction_list:
# 		path = 'data/Human/'+direction+pos+'.dat'
# 		data = np.loadtxt(path)
# 		start_and_end.append([data[0][0],data[1][0],data[0][-1],data[1][-1]])
# print(start_and_end)
# np.savetxt("data/Human/StartAndEnd.dat",start_and_end)

plt.subplot(1,3,1)
m = np.loadtxt("data/Human/S1540.dat")
plt.plot(m[0],m[1],color='green')
plt.plot(m[6],m[7],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[12],m[13],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[18],m[19],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[24],m[25],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[30],m[31],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[36],m[37],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[42],m[43],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[48],m[49],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[54],m[55],color='green',linewidth=0.75,alpha = 0.5)
plt.plot(m[60],m[61],color='green',linewidth=0.75,alpha = 0.5)


arrow_len = 0.1
count = 0
for i in range (len(m[0])):
	if count%10 == 0:
		plt.arrow(m[0][i], m[1][i], np.cos(m[5][i])*arrow_len, np.sin(m[5][i])*arrow_len, head_width=.005)
		if np.sum(m[11]) != 0:
			plt.arrow(m[6][i], m[7][i], np.cos(m[11][i])*arrow_len, np.sin(m[11][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[17]) != 0:		
			plt.arrow(m[12][i], m[13][i], np.cos(m[17][i])*arrow_len, np.sin(m[17][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[23]) != 0:			
			plt.arrow(m[18][i], m[19][i], np.cos(m[23][i])*arrow_len, np.sin(m[23][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[29]) != 0:			
			plt.arrow(m[24][i], m[25][i], np.cos(m[29][i])*arrow_len, np.sin(m[29][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[35]) != 0:			
			plt.arrow(m[30][i], m[31][i], np.cos(m[35][i])*arrow_len, np.sin(m[35][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[41]) != 0:			
			plt.arrow(m[36][i], m[37][i], np.cos(m[41][i])*arrow_len, np.sin(m[41][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[47]) != 0:			
			plt.arrow(m[42][i], m[43][i], np.cos(m[47][i])*arrow_len, np.sin(m[47][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[53]) != 0:			
			plt.arrow(m[48][i], m[49][i], np.cos(m[53][i])*arrow_len, np.sin(m[53][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[59]) != 0:			
			plt.arrow(m[54][i], m[55][i], np.cos(m[59][i])*arrow_len, np.sin(m[59][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		if np.sum(m[65]) != 0:			
			plt.arrow(m[60][i], m[61][i], np.cos(m[65][i])*arrow_len, np.sin(m[65][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
	count += 1


# Speed
plt.subplot(1,3,2)
plt.plot(time,m[0])
plt.plot(time,m[1])
plt.plot(time,m[2])
plt.plot(time,m[3])

# Orientation
plt.subplot(1,3,3)
plt.plot(time,m[4])
plt.plot(time,m[5])

plt.show()

print(len(m[0]))

direction_list = ['N','E','S','O']
position_list = ['1500','4000','-0615','0615','1515','4015','-0640','0640','1540','4040']
path_human_list = []
for pos in position_list:
	for direction in direction_list:
		name_file = direction+pos+".dat"
		path_human_list.append('data/Human/'+name_file)

dist_human_list, ang_dist_human_list= [],[]

for i in range (len(path_human_list)):
	human_data = np.loadtxt(path_human_list[i])		
	for k in range(10):
		ang_dist_human_list.append(angularDistanceBetweenCurvs(human_data[10+6*k],human_data[4]))
		dist_human_list.append(distanceBetweenCurvs(human_data[6+6*k],human_data[0],human_data[7+6*k],human_data[1]))

path = "data/dist_human.dat"
np.savetxt(path,dist_human_list)		

path = "data/ang_dist_human.dat"
np.savetxt(path,ang_dist_human_list)	