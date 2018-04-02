# -*- coding: utf-8 -*-
"""
plot particle cloud at each step in particle filter
@author: atpandey
"""
#%%
import matplotlib.pyplot as plt



part_cloud_file='./outputs/particle_cloud.out'
part_cloud_weights='./outputs/particle_weights.out'
part_cloud_id='./outputs/particle_id.out'

#%%
#PROCESS Particle cloud data
#number ofo steps is number of lines in file
step_counter=0
#number of particles is length of a line in file
#number_of_particles=500
number_of_particles=0
#check that data captured is consistent with number of particles
#if prev particle_number not same as current an error
number_of_particles_prev=0

particle_cloud_map=dict()
step_dict=dict()


with (open(part_cloud_file,'r')) as part_cl:
    
    
    
    for line in part_cl:
        
        line.rstrip()
        line_sp=line.split(';')
        #print("Len:",len(line_sp))
        number_of_particles=len(line_sp)
        if step_counter > 1:
                if number_of_particles_prev!= number_of_particles:
                    print("particle number inconsistent at line:",step_counter) 
                    quit()
        #reassign num parts previous
        number_of_particles_prev=number_of_particles
        
        #the last split field is \n,
        #values are in string type so convert list of string to list of int
        line_sp =list(map(int, line_sp[:-1]))
        step_dict[step_counter]=line_sp
        #step_dict[step_counter]=[int(float(l)) for l in line_sp]
                  
        step_counter +=1
        
    
print ("Number of steps were: ",step_counter)
print ("Number of particles were: ",number_of_particles-1)
#print (np.argmax(step_dict[number_of_particles-1]))
(max_sampled,particle_nr) = max((v,i) for i,v in enumerate(step_dict[step_counter-1]))
#print (m,i)
print ("Max sampled particle was at idx:",particle_nr," value: ",max_sampled)

#%%
#PROCESS Particle weights data
#number ofo steps is number of lines in file
weights_counter=0
#number of particles is length of a line in file
#number_of_particles=500
w_number_of_particles=0
#check that data captured is consistent with number of particles
#if prev particle_number not same as current an error
w_number_of_particles_prev=0

particle_weights_map=dict()
weights_dict=dict()


with (open(part_cloud_weights,'r')) as part_w:
    
    
    
    for line in part_w:
        
        line.rstrip()
        line_sp=line.split(';')
        #print("length:",len(line_sp))
        #print("length:",line_sp[-2:-1])
        w_number_of_particles=len(line_sp)
        if weights_counter > 1:
                if w_number_of_particles_prev!= w_number_of_particles:
                    print("particle number inconsistent at line:",weights_counter) 
                    quit()
        #reassign num parts previous
        w_number_of_particles_prev=w_number_of_particles
        
        #the last split field is \n,
        #values are in string type so convert list of string to list of int
        line_sp =list(map(float, line_sp[:-1]))
        weights_dict[weights_counter]=line_sp
        #step_dict[step_counter]=[int(float(l)) for l in line_sp]
                  
        weights_counter +=1
        
    
print ("Number of steps were: ",weights_counter)
print ("Number of particles were: ",w_number_of_particles-1)
#print (np.argmax(step_dict[number_of_particles-1]))
(wm,wi) = max((wv,wi) for wi,wv in enumerate(weights_dict[weights_counter-1]))
#print (wm,wi)
print ("Max weight for particle was at idx:",wi," value: ",wm)

weights_sum=0
for i in weights_dict[weights_counter-1]:
    weights_sum +=i  
#print (wm,wi)
print ("Sum of weights at last step:",weights_sum)

print("Max sampled particle is:",particle_nr," it's weight is: ",weights_dict[weights_counter-1][particle_nr])

figt, axt = plt.subplots()
x = list(range(number_of_particles-1))
y=weights_dict[weights_counter-1]
axt.scatter(x,y)



#%%
#PROCESS Particle id data
#number ofo steps is number of lines in file
id_counter=0
#number of particles is length of a line in file
#number_of_particles=500
id_number_of_particles=0
#check that data captured is consistent with number of particles
#if prev particle_number not same as current an error
id_number_of_particles_prev=0

particle_id_map=dict()
id_dict=dict()


with (open(part_cloud_id,'r')) as part_id:
    
    
    
    for line in part_id:
        
        line.rstrip()
        line_sp=line.split(';')
        #print("length:",len(line_sp))
        #print("length:",line_sp[-2:-1])
        id_number_of_particles=len(line_sp)
        if id_counter > 1:
                if id_number_of_particles_prev!= id_number_of_particles:
                    print("particle number inconsistent at line:",id_counter) 
                    quit()
        #reassign num parts previous
        id_number_of_particles_prev=id_number_of_particles
        
        #the last split field is \n,
        #values are in string type so convert list of string to list of int
        line_sp =list(map(float, line_sp[:-1]))
        id_dict[id_counter]=line_sp
        #step_dict[step_counter]=[int(float(l)) for l in line_sp]
                  
        id_counter +=1
        
    
print ("Number of steps were: ",id_counter)
print ("Number of particles were: ",id_number_of_particles-1)
#print (np.argmax(step_dict[number_of_particles-1]))
#(idm,idi) = max((idv,idi) for idi,idv in enumerate(id_dict[id_counter-1]))
##print (wm,wi)
#print ("Max weight for particle was at idx:",idi," value: ",idm)

#weights_sum=0
#for i in weights_dict[weights_counter-1]:
#    weights_sum +=i  
#print (wm,wi)
#print ("Sum of weights at last step:",weights_sum)

#print("Max sampled particle is:",particle_nr," it's weight is: ",weights_dict[weights_counter-1][particle_nr])

figt, axt = plt.subplots()
x = list(range(number_of_particles-1))
y=id_dict[id_counter-1]
#y=id_dict[50]
axt.plot(x,y)






#%%
fig0, ax0 = plt.subplots()
ax0.plot(list(range(number_of_particles-1)), step_dict[0], label='step1')
ax0.set_title('PC bar at Step0')
#ax0.plot(list(range(500)), step_dict[0], label='step1')
figs0,axs0=plt.subplots()

x = list(range(number_of_particles-1))
y = [0]*len(x)
s = [1*n for n in step_dict[0]]
axs0.scatter(x,y,s=s)
axs0.set_title('PC at Step0')

fig5, ax5 = plt.subplots()
ax5.plot(list(range(number_of_particles-1)), step_dict[5], label='step5')
ax5.set_title('PC bar at Step5')
figs5,axs5=plt.subplots()
s = [.1*n for n in step_dict[5]]
axs5.scatter(x,y,s=s)
axs5.set_title('PC at Step5')
#fig100, ax100 = plt.subplots()
#ax100.plot(list(range(500)), step_dict[100], label='step100')

figs50,axs50=plt.subplots()
s = [.1*n for n in step_dict[50]]
axs50.scatter(x,y,s=s)
axs50.set_title('PC at Step50')

figslast,axslast=plt.subplots()
s = [.01*n for n in step_dict[step_counter-1] ]
axslast.scatter(x,y,s=s)
axslast.set_title('PC at Last Step')


#%%
fig0, ax0 = plt.subplots()
#for i in range(len(step_dict.keys())):
for i in range(4):    
    x = list(range(number_of_particles-1))
    y = [0]*len(step_dict[i])
    s = [n for n in step_dict[i]]
    ax0.scatter(x,y,s=s)

#%% 
#histogram of particle cloud
##Easier will be only ploy lat step data
#xs=step_counter[len(step_counter.keys())]
xs=range(number_of_particles-1)

#lets normalize particle cloud --> orders of magnitude diff between dominant particle and others
ys_nonnorm=step_dict[step_counter-1]
ys_sum=sum(ys_nonnorm)
#print(ys_sum)
#ys=[i/ys_sum for i in ys_nonnorm]

figh,axh = plt.subplots()
axh.scatter(xs,ys_nonnorm)
axh.set_title('Scatter plot particle sampled')
axh.set_xlabel('Particles')
axh.set_ylabel('Nr times sampled')


#find max sampled partcile and number of times sampled
#that will be at last step in step_dict
(hm,hi) = max((hv,hi) for hi,hv in enumerate(ys_nonnorm))
#print (wm,wi)
print ("Max weight for particle was at idx:",hi," value: ",hm)

txt='{},{}'.format(hi,hm)
axh.annotate(txt, (hi,hm))

#%%
import numpy as np
import matplotlib.animation as animation


figscat,axscat=plt.subplots()

axscat.set_title('Scatter plot of Survived particle at each update Step', fontsize=10)
axscat.set_xlabel('Particles', fontsize=12)
axscat.set_ylabel('Particle Survived', fontsize=12)


scat=plt.scatter([],[],s=10,color='purple')
ann_list=[]
l, = plt.plot([], [], 'rx')
#update set line a value(a list) of dict data
def update_scatter(num ,data,line):
    y = data[num]
    line.set_data(range(len(y)), y)
    
    most_freq=max(set(y), key=y.count)
    ##add and remove annnotation as a scatter plot
    for i ,a in enumerate(ann_list):
        a.remove()
    ann_list[:]=[]
    
    #step and dominant partcile to annotate
    #count of annotations
    n=2
    #set positions for annotation
    scat.set_offsets([(50,50),(150,150)])
    ann=plt.annotate('{} {}'.format('Step: ',num),(50,50),color = "green") 
    ann_list.append(ann)
    ann=plt.annotate('{} {}'.format('Dominant Particle',most_freq),(150,150),color = "blue")
    ann_list.append(ann)
line_scat = animation.FuncAnimation(figscat, update_scatter, len(id_dict.keys()), fargs=(id_dict, l),
                                   interval=20, repeat=True,blit=False)

#line_scat.save('./outputs/particles_develop'+'.gif',writer='imagemagick', fps=50)

plt.rcParams["animation.convert_path"] = "C:\Program Files\ImageMagick-7.0.7-Q8\magick.exe" 

line_scat.save('./outputs/line_particle_survival.gif',writer='imagemagick', extra_args='convert')
plt.xlim(0,500)
plt.ylim(0,500)



#%%
import numpy as np
import matplotlib.animation as animation

#update set line a value(a list) of dict data
def update_line(num ,data, line):
    y = data[num]
    line.set_data(range(len(y)), y)
    #axani.annotate('  ', (num*2,num))
    axani.annotate(num,(num*2,num))
    ann = axani.annotate(num,(num*2,num))
    ann.remove()


    

#figani = plt.figure()

#figani.suptitle('Scatter plot Survived partcile at each update Step', fontsize=10)
#plt.xlabel('Particles', fontsize=15)
#plt.ylabel('Particle Survived', fontsize=15)

figani,axani = plt.subplots()

axani.set_title('Scatter plot of Survived particle at each update Step', fontsize=10)
axani.set_xlabel('Particles', fontsize=12)
axani.set_ylabel('Particle Survived', fontsize=12)


l, = plt.plot([], [], 'rx')
#l = plt.scatter([], [], s=10)
line_ani = animation.FuncAnimation(figani, update_line, len(id_dict.keys()), fargs=(id_dict, l),
                                   interval=500, repeat=True,blit=False)


plt.xlim(0,500)
plt.ylim(0,500)