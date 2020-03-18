import numpy as np
import matplotlib.pylab as plt
from math import pi,floor,sqrt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

dist_clotho = np.transpose(np.loadtxt("data/dist_clotho.dat"))
dist_ddp = np.transpose(np.loadtxt("data/dist_ddp.dat"))
print(len(dist_clotho))
print("mean dist clotho :", np.sum(dist_clotho)/40,"mean dist OC :",np.sum(dist_ddp)/40)
print("mean dist clotho (pi/2):", np.sum([dist_clotho[i] for i in range(0,len(dist_clotho)-3,4)])/10,\
"mean dist OC (pi/2):",np.sum([dist_ddp[i] for i in range(0,len(dist_ddp)-3,4)])/10)
print("mean dist clotho (0):", np.sum([dist_clotho[i] for i in range(1,len(dist_clotho)+1-3,4)])/10,\
"mean dist OC (0):", np.sum([dist_ddp[i] for i in range(1,len(dist_ddp)+1-3,4)])/10)
print("mean dist clotho (-pi/2):", np.sum([dist_clotho[i] for i in range(2,len(dist_clotho)+2-3,4)])/10,\
"mean dist OC (-pi/2):", np.sum([dist_ddp[i] for i in range(2,len(dist_ddp)+2-3,4)])/10)
print("mean dist clotho (pi):",np.sum([dist_clotho[i] for i in range(3,len(dist_clotho)+3-3,4)])/10,\
"mean dist OC (pi):", np.sum([dist_ddp[i] for i in range(3,len(dist_ddp)+3-3,4)])/10)

distances = [abs(0.6-1.5),abs(0.6-4),sqrt((-0.6-0.6)**2+1.5**2),1.5,\
sqrt((1.5-0.6)**2+1.5**2),sqrt((4-0.6)**2+1.5**2),sqrt((-0.6-0.6)**2+4**2),\
4,sqrt((1.5-0.6)**2+4**2),sqrt((4-0.6)**2+4**2)]
print(distances)

orientations = ["pi/2 OC","pi/2 cloth","0 OC","0 cloth","-pi/2 OC","-pi/2 cloth","pi OC","pi cloth"]
colors = ['red','green','blue','black']

for i in range (0,40,4):
	mean_ddp = 0
	mean_cloth = 0
	for j in range (4):
		mean_ddp += dist_ddp[i+j]
		mean_cloth += dist_clotho[i+j] 
		plt.scatter(distances[i/4], dist_ddp[i+j], marker='^', color = colors[j])
		plt.scatter(distances[i/4], dist_clotho[i+j], marker='o', color = colors[j])
	#print(i,mean_ddp/4,mean_cloth/4)
plt.legend(orientations,loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=4, fancybox=True)
plt.ylabel("distance to the mean trajectory (m)")
plt.xlabel("distance to the goal (m)")
plt.show()

# orientation = ["pi/2","0","-pi/2","pi"]
# for count in range (4):
# 	plt.subplot(1,4,count+1)
# 	plt.plot(np.arange(1,11,1),[dist_clotho[i] for i in range(count,len(dist_clotho)+count-3,4)])
# 	plt.plot(np.arange(1,11,1),[dist_ddp[i] for i in range(count,len(dist_ddp)+count-3,4)])
# 	plt.axis([1,10,0,3.1])
# 	plt.title(orientation[count])
# plt.show()

plt.subplot(1,2,1)
plt.boxplot([dist_clotho])
# plt.ylim(0, 14)
plt.title('Clothoid')
plt.subplot(1,2,2)
plt.boxplot([dist_ddp])
# plt.ylim(0, 14)
plt.title('Optimal Control')
plt.show()

