import numpy as np
import matplotlib.pylab as plt
from math import pi,floor,sqrt
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

dist_ddp = np.transpose(np.loadtxt("data/dist_ddp.dat"))
dist_fin_ddp = np.transpose(np.loadtxt("data/dist_fin_ddp.dat"))

angular_dist_ddp = np.transpose(np.loadtxt("data/angular_dist_ddp.dat"))
angular_dist_fin_ddp = np.transpose(np.loadtxt("data/angular_dist_fin_ddp.dat"))

dist_human = np.transpose(np.loadtxt("data/dist_human.dat"))
angular_dist_human = np.transpose(np.loadtxt("data/ang_dist_human.dat"))

dist_subject_ddp = np.loadtxt("data/dist_subjects_ddp.dat")

distances = [abs(0.6-1.5),abs(0.6-4),sqrt((-0.6-0.6)**2+1.5**2),1.5,\
sqrt((1.5-0.6)**2+1.5**2),sqrt((4-0.6)**2+1.5**2),sqrt((-0.6-0.6)**2+4**2),\
4,sqrt((1.5-0.6)**2+4**2),sqrt((4-0.6)**2+4**2)]
print(distances)

print("-------------------------------------------------------------------")

print("mean dist human :",np.mean(dist_human),"std dist human :",np.std(dist_human))
print("mean angular dist human :",np.mean(angular_dist_human),"std dist human :",np.std(angular_dist_human))
print("max linear dist human :",np.max(dist_human),"max angular dist human :",np.max(angular_dist_human))

print("-------------------------------------------------------------------")

print("max dist subjects/ddp :",np.max(dist_subject_ddp),"std dist subjects/ddp :",np.min(dist_subject_ddp))
print("mean dist subjects/ddp :",np.mean(dist_subject_ddp),"std dist subjects/ddp :",np.std(dist_subject_ddp))


print("-------------------------------------------------------------------")

print("mean final dist OC : ",np.mean(dist_fin_ddp))

print("mean dist OC :",np.mean(dist_ddp),"std dist OC :",np.std(dist_ddp))

print("mean dist OC (pi/2):",np.mean([dist_ddp[i] for i in range(0,len(dist_ddp)-3,4)]))
print("mean dist OC (0):", np.mean([dist_ddp[i] for i in range(1,len(dist_ddp)+1-3,4)]))
print("mean dist OC (-pi/2):", np.mean([dist_ddp[i] for i in range(2,len(dist_ddp)+2-3,4)]))
print("mean dist OC (pi):", np.mean([dist_ddp[i] for i in range(3,len(dist_ddp)+3-3,4)]))

ind = np.where(angular_dist_ddp != 0)

print("mean final angular dist OC : ",np.mean(angular_dist_fin_ddp[ind]))

print("mean angular dist OC :",np.mean(angular_dist_ddp[ind]),"std dist OC :",np.std(angular_dist_ddp[ind]))

print("mean angular dist OC (pi/2):",np.mean([angular_dist_ddp[i] for i in range(0,len(angular_dist_ddp)-3,4)]))
print("mean angular dist OC (0):", np.mean([angular_dist_ddp[i] for i in range(1,len(angular_dist_ddp)+1-3,4)]))
print("mean angular dist OC (-pi/2):", np.mean([angular_dist_ddp[i] for i in range(2,len(angular_dist_ddp)+2-3,4)]))

ang_ddp = np.array([angular_dist_ddp[i] for i in range(3,len(angular_dist_ddp)+3-3,4)])
ind =  np.where(ang_ddp != 0)

print("mean angular dist OC (pi):", np.mean(ang_ddp[ind]))

print("-------------------------------------------------------------------")

KS_test_lin = stats.kstest(dist_ddp,'norm', args=(np.mean(dist_ddp), np.std(dist_ddp)))
KS_test_ang = stats.kstest(angular_dist_ddp[ind],'norm', args=(np.mean(angular_dist_ddp), np.std(angular_dist_ddp)))

print('Kolmogorov-Smirnov test - Not normal if p < 0.05')
print("p for linear dist: ",KS_test_lin)
print("p for angular dist: ",KS_test_ang)

KS_test_human = stats.kstest(dist_human,'norm', args=(np.mean(dist_human), np.std(dist_human)))

print("p for human dist: ",KS_test_human)

anova_human_ddp = stats.f_oneway(dist_human,dist_ddp)

student_human_ddp = stats.ttest_ind(dist_human,dist_ddp) ;

print('Significant difference if p < 0.05 - Simu/Mes')
print("ANOVA test, p : ",anova_human_ddp)
print("STUDENT test, p : ",student_human_ddp)

# PLOT dist/dist ######################################################################
print("-------------------------------------------------------------------")


orientations = ["pi/2 rad","0 rad","-pi/2 rad","pi rad"]
colors = ['red','green','blue','black']

for i in range (0,40,4):
	mean_ddp = 0
	for j in range (4):
		mean_ddp += dist_ddp[i+j]
		# plt.scatter(distances[i/4], dist_ddp[i+j], marker='o', color = colors[j])
	# print(i,mean_ddp/4)
# plt.legend(orientations,loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, fancybox=True)
# plt.ylabel("distance to the mean trajectory (m)")
# plt.xlabel("distance to the goal (m)")
# plt.show()

for i in range (0,40,4):
	mean_ddp = 0
	count_ddp = 0
	for j in range (4):
		mean_ddp += angular_dist_ddp[i+j]
		if angular_dist_ddp[i+j]!=0:
			count_ddp += 1		
		# print(angular_dist_ddp[i+j])

		# plt.scatter(distances[i/4], angular_dist_ddp[i+j], marker='o', color = colors[j])
		# print(i,mean_ddp/count_ddp)
# plt.legend(orientations,loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, fancybox=True)
# plt.ylabel("distance to the mean trajectory (rad)")
# plt.xlabel("distance to the goal (m)")
# plt.show()

# dist according to orientation ##################################################
print("-------------------------------------------------------------------")

dist_ddp_ori = np.zeros((4,10))
ang_dist_ddp_ori = np.zeros((4,10))

orientation = ["pi/2","0","-pi/2","pi"]
for count in range (4):
	ang_dist_ddp_ori[count] = [angular_dist_ddp[i] for i in range(count,len(angular_dist_ddp)+count-3,4)]
	dist_ddp_ori[count] = [dist_ddp[i] for i in range(count,len(dist_ddp)+count-3,4)]
	# plt.subplot(1,4,count+1)
	# plt.plot(np.arange(1,11,1),[dist_ddp[i] for i in range(count,len(dist_ddp)+count-3,4)])
	# plt.axis([1,10,0,0.2])
	# plt.title(orientation[count])
# plt.show()

anova_lin = stats.f_oneway(dist_ddp_ori[0],dist_ddp_ori[1],dist_ddp_ori[2],dist_ddp_ori[3])
anova_ang = stats.f_oneway(ang_dist_ddp_ori[0],ang_dist_ddp_ori[1],ang_dist_ddp_ori[2],ang_dist_ddp_ori[3])

print('ANOVA test - Significant difference if p < 0.05 - Orientation/Orientation')
print("p for linear dist: ",anova_lin)
print("p for angular dist: ",anova_ang)

# print('Student test - Significant difference if p < 0.05')
# for i in range(4):
# 	print(stats.ttest_ind(dist_ddp_ori[0],dist_ddp_ori[i]))

# dist according to dist ##################################################
print("-------------------------------------------------------------------")

dist_ddp_sup,dist_ddp_inf = [],[]
dist_ang_ddp_sup,dist_ang_ddp_inf = [],[]
for i in range (0,40,4):
	for j in range (4):
		if distances[i/4] > 3:
			dist_ddp_sup.append(dist_ddp[i+j])
			dist_ang_ddp_sup.append(angular_dist_ddp[i+j])
		else:
			dist_ddp_inf.append(dist_ddp[i+j])
			dist_ang_ddp_inf.append(angular_dist_ddp[i+j])

anova_dist = stats.f_oneway(dist_ddp_inf,dist_ddp_sup)
anova_dist_ang = stats.f_oneway(dist_ang_ddp_inf,dist_ang_ddp_sup)

print('ANOVA test - Significant difference if p < 0.05 - Distance/Distance')
print("p for linear dist: ",anova_dist)
print("p for angular dist: ",anova_dist_ang)

anova_dist_ori = stats.f_oneway(dist_ddp_inf,dist_ddp_sup,dist_ddp_ori[0],\
	dist_ddp_ori[1],dist_ddp_ori[2],dist_ddp_ori[3])
anova_dist_ori_ang = stats.f_oneway(dist_ang_ddp_inf,dist_ang_ddp_sup,ang_dist_ddp_ori[0],\
	ang_dist_ddp_ori[1],ang_dist_ddp_ori[2],ang_dist_ddp_ori[3])

print('ANOVA test - Significant difference if p < 0.05 - Distance/Orientation')
print("p for linear dist: ",anova_dist_ori)
print("p for angular dist: ",anova_dist_ori_ang)


# dist according to subjects ##################################################
print("-------------------------------------------------------------------")

dist_human_sub, angular_dist_human_sub = np.zeros((10,40)),np.zeros((10,40))
for i in range (0,40):
	for j in range (10):
		dist_human_sub[j][i] = dist_human[j+10*i]
		angular_dist_human_sub[j][i] = angular_dist_human[j+10*i]
		
anova_dist = stats.f_oneway(dist_human_sub[0],dist_human_sub[1],dist_human_sub[2],\
	dist_human_sub[3],dist_human_sub[4],dist_human_sub[5],dist_human_sub[6],\
	dist_human_sub[7],dist_human_sub[8],dist_human_sub[9])
anova_dist_ang = stats.f_oneway(angular_dist_human_sub[0],angular_dist_human_sub[1],angular_dist_human_sub[2],\
	angular_dist_human_sub[3],angular_dist_human_sub[4],angular_dist_human_sub[5],angular_dist_human_sub[6],\
	angular_dist_human_sub[7],angular_dist_human_sub[8],angular_dist_human_sub[9])

print('ANOVA test - Significant difference if p < 0.05 - Subjects/Human')
print("p for linear dist: ",anova_dist)
print("p for angular dist: ",anova_dist_ang)


# BOX PLOT ######################################################################
print("-------------------------------------------------------------------")

# ind = np.where(angular_dist_ddp != 0)
# # plt.subplot(1,3,1)
# # plt.boxplot([dist_human])
# # plt.ylim(0, 0.2)
# # plt.title('Human')
# plt.subplot(1,2,1)
# plt.boxplot([dist_ddp])
# plt.ylim(0, 0.2)
# # plt.ylim(0, 14)
# plt.title('d_xy (m)')
# plt.subplot(1,2,2)
# plt.boxplot([angular_dist_ddp[ind]])
# # plt.ylim(0, 14)
# plt.title('d_theta (rad)')

# plt.show()

# plt.boxplot([dist_human])
# plt.title('d_xy (m) human')

# plt.show()

# plt.subplot(2,2,1)
# plt.boxplot([dist_ddp_ori[0]])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), theta_0 = pi/2 rad)')
# plt.subplot(2,2,2)
# plt.boxplot([dist_ddp_ori[1]])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), theta_0 = 0 rad')
# plt.subplot(2,2,3)
# plt.boxplot([dist_ddp_ori[2]])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), theta_0 = -pi/2 rad')
# plt.subplot(2,2,4)
# plt.boxplot([dist_ddp_ori[3]])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), theta_0 = pi rad')
# plt.show()


# plt.subplot(1,2,1)
# plt.boxplot([dist_ddp_inf])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), d < 3 m')
# plt.subplot(1,2,2)
# plt.boxplot([dist_ddp_sup])
# plt.ylim(0, 0.2)
# plt.title('d_xy (m), d > 3 m')

# plt.show()