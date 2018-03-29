# -*- coding: utf-8 -*-
"""
plot particle cloud at each step in particle filter
@author: atpandey
"""
#%%
import matplotlib.pyplot as plt



part_cloud_file='./outputs/particle_cloud.out'

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

#print (np.argmax(step_dict[number_of_particles-1]))
(m,i) = max((v,i) for i,v in enumerate(step_dict[step_counter-1]))
print (m,i)
print ("Max sampled particle was at idx:",i," value: ",m)


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

figs100,axs100=plt.subplots()
s = [.1*n for n in step_dict[100]]
axs100.scatter(x,y,s=s)
axs100.set_title('PC at Step100')

figslast,axslast=plt.subplots()
s = [.01*n for n in step_dict[step_counter-1] ]
axslast.scatter(x,y,s=s)
axslast.set_title('PC at Last Step')


#%%
fig0, ax0 = plt.subplots()
for i in range(len(step_dict.keys())):
    x = list(range(number_of_particles-1))
    y = [0]*len(step_dict[i])
    s = [1*n for n in step_dict[0]]
    axs0.scatter(x,y,s=s)

#%% 
#histogram of particle cloud
##Easier will be only ploy lat step data
#xs=step_counter[len(step_counter.keys())]
xs=range(number_of_particles-1)

#lets normalize particle cloud --> orders of magnitude diff between dominant particle and others
ys_nonnorm=step_dict[step_counter-1]
ys_sum=sum(ys_nonnorm)
print(ys_sum)
ys=[i/ys_sum for i in ys_nonnorm]

fig = plt.figure()
plt.scatter(xs,ys_nonnorm)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ffig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 50000))
#fig, ax = plt.subplots()
ax.grid('on')
#line, = ax.plot([], [], lw=2)
line, = ax.plot([], [], 'ro')

def animate(i):
    #line.set_ydata(np.sin(x + i/10.0))  # update the data
    y=[n for n in step_dict[i]]
    line.set_data(range(len(y)), y)
    return line,

# Init only required for blitting to give a clean slate.
def init():
    #line.set_ydata(np.ma.array(x, mask=True))
    #line.set_data(range(500), [0 for _ in range(500)])
    line.set_data([], [])
    return line,

#ani = animation.FuncAnimation(fig, animate, np.arange(1, 100), init_func=init,
#                              interval=25, blit=True)

#ani = animation.FuncAnimation(fig, animate, init_func=init,frames=200,
#                              interval=250, blit=True)
ani = animation.FuncAnimation(fig, animate, init_func=init,frames=step_dict.keys(),
                              interval=500, blit=True)


plt.show() 


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ffig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 10000))
ax.grid('on')
#line, = ax.plot([], [], lw=2)
line, = ax.plot([], [], 'ro')
def update_line(num ,data, line):
    y = data[num*0.1]
    #line.set_data(range(len(y)), y)

def animate(i):
    #line.set_ydata(np.sin(x + i/10.0))  # update the data
    y=step_dict[i]
    line.set_data(range(len(y)), y)
    return line,


# Init only required for blitting to give a clean slate.
def init():
    #line.set_ydata(np.ma.array(x, mask=True))
    #line.set_data(range(500), [0 for _ in range(500)])
    line.set_data([], [])
    return line,

#ani = animation.FuncAnimation(fig, animate, np.arange(1, 100), init_func=init,
#                              interval=25, blit=True)

ani = animation.FuncAnimation(fig, animate, init_func=init,frames=200,
                              interval=250, blit=True)

plt.show() 
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show() 


#%%



#%%
#print("Keys",step_dict.keys()) 
#for i in step_dict.keys():
    #print(step_dict[i])
    
##########################################
# display scatter plot data
#plt.figure(figsize=(10,8))
#plt.title('Scatter Plot', fontsize=20)
#plt.xlabel('step count', fontsize=15)
#plt.ylabel('partciles', fontsize=15)
#plt.scatter(step_dict.keys[], step_dict.values, marker = 'o')
#####################################

#//
#xs, ys=zip(*((x, k) for k in step_dict.keys() for x in step_dict[k]))
#plt.plot(ys, xs, 'ro')
#//



##############
#import itertools
#x= []
#y= []
#for k, v in step_dict.iteritems():
#    x.extend(list(itertools.repeat(k, len(v))))
#    y.extend(v)
#plt.xlim(0,5)
#plt.plot(x,y,'ro')
###############################



## add labels
#for label, x, y in zip(data["label"], data["x"], data["y"]):
#    plt.annotate(label, xy = (x, y))
#%%


#from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = plt.figure()
ax = Axes3D(fig)

#sequence_containing_x_vals = list(range(0, 100))
#sequence_containing_y_vals = list(range(0, 100))
#sequence_containing_z_vals = list(range(0, 100))

#random.shuffle(sequence_containing_x_vals)
#random.shuffle(sequence_containing_y_vals)
#random.shuffle(sequence_containing_z_vals)

#ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
xs=list(step_dict.keys())
for i in range(len(step_dict.keys())):
    
    ys=range(len(step_dict[i]))
    zs=step_dict[i]
    ax.scatter(xs, ys, zs)
    #ax.scatter(xs, ys)


plt.show()




#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zlow, zhigh)
#    ax.scatter(xs, ys, zs, c=c, marker=m)

xs=list(step_dict.keys())
for i in range(len(step_dict.keys())):
    
    ys=range(len(step_dict[i]))
    zs=step_dict[i]
    ax.scatter(xs, ys, zs)
    #ax.scatter(xs, ys)
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

plt.show()



#%%

#for key in step_dict:
#
#    plt.scatter([key]*len(step_dict[key]), step_dict[key], label=key)
#
#plt.legend()
#plt.show()

array = np.zeros((8,8))
for key in comp:
    array[:,key] = comp[key]

x = range(8)
for i in range (8):
    plt.scatter(x, array[i,:], label=i)

plt.legend()
plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)


def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)


plt.show()




#%%
import numpy as np
import matplotlib.animation as animation

#update set line a value(a list) of dict data
def update_line(num ,data, line):
    y = data[num*0.1]
    line.set_data(range(len(y)), y)


    

fig1 = plt.figure()

# just creat a dictionary of lists by variable length
data = {}
for i in range(20):
    t = i*0.1
    data[t] = [j*.5 for j in range(np.random.randint(2,5))]
    #t=i
    #data[t] = [j for j in range(4)]

l, = plt.plot([], [], 'rx')
line_ani = animation.FuncAnimation(fig1, update_line, 20, fargs=(data, l),
                                   interval=500, repeat=True)



plt.xlim(0,5)
plt.ylim(0,2.5)
#plt.xlim(0,20)
#plt.ylim(0,5)
plt.show()  