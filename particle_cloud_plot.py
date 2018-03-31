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
#y=id_dict[id_counter-1]
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
print(ys_sum)
ys=[i/ys_sum for i in ys_nonnorm]

fig = plt.figure()
plt.scatter(xs,ys_nonnorm)

#%%
import numpy as np
import matplotlib.animation as animation

#update set line a value(a list) of dict data
def update_line(num ,data, line):
    y = data[num]
    line.set_data(range(len(y)), y)


    

fig1 = plt.figure()

# just creat a dictionary of lists by variable length
data = {}
for i in range(len(id_dict.keys())):
    t = i
    data[t] = [j for j in id_dict[i]]
    #t=i
    #data[t] = [j for j in range(4)]

l, = plt.plot([], [], 'rx')
line_ani = animation.FuncAnimation(fig1, update_line, 500, fargs=(data, l),
                                   interval=100, repeat=True)



plt.xlim(0,500)
plt.ylim(0,500)
#plt.xlim(0,20)
#plt.ylim(0,5)
plt.show()  




#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ffig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 500))
#fig, ax = plt.subplots()
ax.grid('on')
#line, = ax.plot([], [], lw=2)
line, = ax.plot([], [], 'ro')

def animate(i):
    #line.set_ydata(np.sin(x + i/10.0))  # update the data
    #y=[n for n in step_dict[i]]
    y=[n for n in id_dict[i]]
    
    line.set_data(range(len(y)), y)
    print("step",i)
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
ani = animation.FuncAnimation(fig, animate, init_func=init,,
                              interval=500, blit=True)

#ani=FuncAnimation(fig, animate, frames=len(id_dict.keys()),interval=250, repeat=True)

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

#%%
w_l=[0.015858477979868547, 1.8478689417888968e-09, 2.320392111186335e-06, 2.30090491979373, 8.488668402050235, 0.06753863395879998, 16.973028946763502, 0.00797942162393479, 1.1485350532411653, 2.9749256565738447e-09, 0.13211890295591036, 8.742683859566394, 8.719962473439487, 0.02434662401454689, 3.2638833802519707, 2.3077395677510397, 0.6751676580897507, 0.3320597754787183, 2.090308228272102, 0.18241691757936343, 0.5793832273692695, 16.803863445493985, 2.639450813635429e-06, 3.2857070399485537, 0.8908924401265446, 3.493265495465089, 0.0028314439621415837, 0.8483684969598194, 1.7246253715607447, 0.09853079832422836, 8.193590721207023e-06, 1.5851326605498972, 0.41744856772199224, 0.5169972320296826, 1.3139432375262232, 0.044527525312683953, 0.22626006208291313, 3.965023942818313, 11.398195361828236, 2.8917602062426035, 5.988797408802162, 0.00034425280158561036, 0.07072189006983866, 0.395994406080108, 0.359417187163136, 0.21094752315898657, 1.191770718238454, 0.01502592024090577, 0.1265693504290133, 0.2397800327634614, 0.029434725582581093, 0.15872394941508564, 4.449734617069523, 6.661503678566207e-06, 0.011647832379420523, 9.732900767527447, 0.0005121996976950801, 0.38863898115438245, 5.30850596160928, 0.0216780935229514, 0.004073713345519002, 7.820043268346706, 0.12604549053080985, 0.3247471479740985, 5.266038386304155, 0.35993979042412433, 1.3068061495783126e-08, 2.2944141518858596, 0.2498390075511143, 0.102264774617117, 2.201967460031027, 0.10878310327583833, 3.0273035700032755, 0.1487766751831051, 0.17254019444893867, 0.0003501617930521415, 3.6062067631421164, 5.379863810617599, 0.00020917367376843698, 0.10941657167885172, 7.865073552053921, 0.03706369456358927, 5.782377804449097, 1.4118823120997763, 0.07976454379085415, 2.211404153010437, 0.00016461960821637233, 0.030944591093636807, 0.4177908305822253, 0.044986783897372505, 1.246485616091375, 0.507667641462556, 0.11554924586218299, 0.6547191065656878, 1.9836761216435224, 0.5203655147124796, 0.0141597005610081, 14.786925703446691, 1.7494516217518327e-10, 0.01962016838543562, 0.0774621214501509, 4.534112952136681, 0.2818897895088435, 4.131204222639594, 5.02558761222175, 3.413066973114951, 0.007776754663688646, 19.178634149448154, 1.898470453322652, 4.432744174442161, 0.005663552219377467, 0.1953408091514828, 4.164267352863208e-08, 8.406654331287173, 8.149614746680315, 0.003596752070446697, 0.101446899027733, 0.03870231924098901, 8.230400439899176, 0.009512654270069807, 1.0943778799914607, 0.008254742452798263, 0.31448705280054773, 6.3954949193290656, 4.657665711509488, 0.5773231995968191, 0.042093975911266575, 0.008220790220424724, 1.163402890523295, 1.2183658411777014, 0.0029913264736853877, 3.3284498825850695, 0.038830849989195795, 3.5164261002504746, 3.9730057495293254, 0.0064015262858683995, 0.6462156846878425, 0.007899967050144402, 0.03955536278962917, 0.00027608174655706754, 0.1938520815923333, 0.24346778885796566, 0.8383716929546178, 3.4258588546521422e-06, 0.00016042437974787864, 2.332524287817667, 14.482320106013484, 5.016227624079401, 0.7209575736165095, 10.77281247136793, 0.4484705612645386, 0.14956854390693095, 0.04343821898174027, 0.05391220818078432, 4.232000695614803, 0.4137162809902943, 4.577409365999441, 3.5926310579492444, 6.953320183549989e-05, 0.01869146130940111, 0.009155534289679586, 1.682905638092373, 0.0037343209202233686, 0.0073253017350904705, 7.403482029517123, 5.6289962805823865, 5.586759762727777, 9.896398395022746e-05, 0.4189166322330024, 0.0046915080905331205, 15.68864116127396, 0.09265735531667166, 0.0017448681075978578, 0.22749400894966917, 0.050316289033461525, 5.221318584339489, 8.975653399453297, 13.617106133609136, 3.604428646906506, 0.4727704374275832, 0.022699167125867937, 2.759790450300036, 0.02478041488424283, 0.12855286678969285, 1.8342292252093575, 2.5089954500087885, 2.085011844212808, 4.825119996105186, 0.6539315759966392, 12.269956390576176, 1.0151277088202149, 1.1408511328168468e-05, 2.7240038650534926e-24, 0.039975398397313365, 9.190797111038237e-05, 4.087866745937787e-06, 7.579058168715065, 9.47393035315228, 0.014961183552962701, 0.9668046650462897, 1.3621098939762684, 3.846167117127146, 1.3283906085271298, 0.04157541332032552, 2.8790059877558107, 0.9764481345687688, 0.016221882128430818, 0.7635343970568254, 4.866903049585462, 20.493283972659945, 1.4684704749679383, 0.1719105162137988, 0.5098715340257315, 0.0048844043468744365, 3.791208203287736, 4.110644024537537, 0.66948498956449, 0.9055074368038203, 0.8249222510347078, 1.5199795142604464, 0.00036673787879799884, 0.22171114432271746, 9.502750408271153e-14, 0.050067984049409056, 0.7367417457637785, 1.6199623841337258, 1.0656858824237216, 0.14276087435210338, 0.00027471246522417736, 0.1374884732290643, 0.919531250007051, 5.901559317654204, 1.4235315393904926, 2.6263506488470827, 3.9100238171883097, 4.507809218365414, 4.538703058940146, 0.9465385989576721, 19.553947498616775, 0.41444855935510777, 0.027680377694213432, 0.00019427290845871887, 0.0015279263773130055, 0.03495168727019455, 0.006096838465214, 0.27050001304129623, 2.1938059560258836, 0.012891012141568238, 2.789656091511688, 6.813062574312999, 7.7262021116467725, 9.286619466491965, 0.00015584532133416029, 1.3578297387690206, 0.1777893042282782, 1.3869907132599884, 0.004185829849832571, 0.007288387819313045, 5.5670950634607035, 11.22338908884833, 3.6972722450714706, 4.863196714578434, 0.8581266955358829, 3.3844305234710816, 1.7364344871287605, 0.2060546116765547, 0.19200918991531715, 0.24437299601852122, 0.015396831557163534, 6.433245079789956, 0.04673576393944615, 4.672022506420852, 0.16400534640858128, 1.4856584757786606, 1.1090835713686447e-07, 1.4765416408864591, 5.227506372890642, 11.823171345225617, 2.4728679992067217, 3.677087289228858, 0.04773884308263717, 1.251428685069789, 0.6447378518231853, 5.286293398020071, 1.2751517068309062, 0.006273329697900142, 0.00616667515617278, 4.864424113276182, 2.215605445878208, 4.108373906737572, 0.5620336308178803, 0.03207784640953414, 0.48186308904690067, 0.10358666069410076, 1.1415240393759483, 0.0017478117215850232, 1.984786252001852, 0.030585301353342836, 1.3564852014452886, 0.0015724672037358187, 5.95316365785251e-18, 5.030465354296373, 3.6665031251450206, 1.006618164651553, 10.589025990558474, 0.0007552927900158863, 0.32603376118208705, 6.025720896689103, 3.5966521532879, 2.8124332608334203e-08, 0.012126102911520774, 10.050914905006948, 0.001916405180465195, 3.8553835421283917, 1.4966462776368259e-05, 0.001238897668062044, 2.150053067905795, 0.0069959343587784216, 2.338350133529246e-08, 2.399668274446396, 0.08327399413827731, 3.5875462988362807e-06, 0.5782860030734992, 0.13227172683898566, 1.7445638726143218, 0.5683178058940044, 11.585705566284108, 0.9915266977707214, 0.24523672996756932, 0.014567989791745343, 0.03217350581291163, 12.707533989479478, 0.7525496754638561, 2.25096125627402e-05, 1.8282644192542197, 2.0030254532145033, 9.242942337456398, 1.246199354559952, 0.0013820468977178743, 0.1477568991231477, 0.005227978876222712, 3.5939678971349154, 0.20764558115474877, 0.22812431730644334, 0.0008544263136328222, 4.658916574252769, 6.0628226134248955, 1.389374256713804, 0.0001875446384166368, 2.5533156304580875e-05, 0.01205339668561982, 1.460239358791354e-05, 9.861269831306284, 10.009788515007473, 4.9314659854853415, 14.174674893049206, 1.7342741486741056, 15.243209263586905, 0.037125895719228905, 7.405746422016155, 8.931706099686264e-09, 1.9194942764179066, 0.012682682116043687, 0.2589250430187023, 1.0627859387908674, 0.34022710232629044, 0.061334500753273964, 0.5441056867658707, 0.619656343057079, 4.088222527026477, 7.959894849131209, 1.0708778854094982, 2.7181979889816494, 0.1426830971721676, 1.8003777975342872, 1.3234182555483094, 1.2965362185470815, 0.002577733149715848, 0.0036119518437734066, 0.0022492670792585686, 0.49395605137221493, 0.6409174021650178, 0.4331522228980731, 0.4675639007947176, 2.7355130039144613, 0.1428132736885426, 0.07996727810388501, 0.0257180457639815, 1.117044477207618, 0.13052868313679794, 2.9022296187595202, 2.7231351035030176e-13, 22.408720043634684, 3.8551939696648443, 0.03915265977998468, 0.008444285236232333, 14.165305417897182, 5.358650379528223, 11.523936014216785, 0.3152218587445909, 0.019112666336717158, 4.673622993804598e-06, 0.3974729062774586, 8.567272914401048, 5.609190385854925e-07, 0.08427357041976885, 0.23564143013970856, 6.1752061617930885, 3.1947462247503897, 17.459284196932607, 16.98661912559709, 2.0976705641057913, 0.273966844672097, 0.006346012903967093, 1.5800389034446725, 0.0675837196718592, 17.717521903657627, 0.5514948698552131, 0.13493163205637948, 0.0016834262385850664, 5.873378666235304, 0.9555180701397357, 2.0012160008368114, 1.4042680344850713, 0.012028427802261963, 6.551531381850884e-07, 0.6006676323149298, 0.6449150837735751, 3.347878493740249, 0.02221586702878503, 0.034379012495100725, 9.496960058034386, 0.2381910649382456, 3.9528503053938826e-11, 5.801710098596623, 1.66548121465733, 0.13656142400070426, 4.970630204637144, 0.009173193627048274, 1.2282686509073895, 0.06750488328654175, 20.72960262873547, 4.5134779812998245, 4.295109151296684, 0.027807813894598875, 0.03795787048394008, 11.263380943603735, 0.9839567988263103, 1.4799396884612501, 0.013027880632090944, 0.15576975217828326, 0.5904391897465429, 23.62064063712063, 10.649279520303203, 1.1408590253783115, 0.002832346242032366, 0.007733844787136087, 12.19266277967599, 0.1775192222446675, 0.16031800298035861, 1.9346209440322806, 1.821438235555326e-10, 3.1272563476466706, 0.001750735199678906, 0.04896467651771723, 2.621271705191441, 0.23129282194713582, 0.45992538133787053, 3.586209384944894, 1.8414361699467585, 35.08073974257975, 0.09058809673198986, 0.6590649636868635, 0.09425809773771378, 4.32847307175015, 15.087575344397996, 0.13853049810967746, 1.3425263896986785e-05, 2.552506334099693e-09, 0.06696860849824555, 2.240301357595755, 4.567395341276451, 5.172595254155066, 4.496178533606345, 1.9409935655369096e-17, 0.09648691255928155, 0.1050967850884958, 7.324206307111053e-17, 2.034496995574341, 0.43087073550274535, 25.36900899395911, 7.789415563403703e-05, 1.8386232407394374, 0.23654886270915235, 5.454779561971673e-07, 7.640298488878455, 3.8853679785849273, 53.01653055010323, 1.3126901640615585, 0.5622410213978255]
ff,axp=plt.subplots()
x=range(len(w_l))
y=w_l
axp.scatter(x,y)