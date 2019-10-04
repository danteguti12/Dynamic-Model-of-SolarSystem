# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:45:41 2019

@author: Dante
"""

import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import math

#Universal constant
G=6.67408e-11 #Nm2/kg2

#physical quantities
m_nd=1.989e+30 #mass of the sun kg
r_nd=7.785e+11 #mean distance between sun and jupiter
v_nd=13100 #velocity of jupiter with respect to the sun
t_nd=11.86*365*24*3600 #orbital period of jupiter

#net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#keplerian elements to cartesian
def kep_car(p,e,i,om,raan,u,v):
    
    i=i*math.pi/180
    om=om*math.pi/180
    raan=raan*math.pi/180
    v=v*math.pi/180
        
    x=(p*math.cos(v)/(1+e*math.cos(v)))*(math.cos(raan)*math.cos(om)-math.sin(raan)*math.sin(om)*math.cos(i))+ \
    (p*math.sin(v)/(1+e*math.cos(v)))*(-math.cos(raan)*math.sin(om)-math.sin(raan)*math.cos(om)*math.cos(i))
    
    y=(p*math.cos(v)/(1+e*math.cos(v)))*(math.sin(raan)*math.cos(om)+math.cos(raan)*math.sin(om)*math.cos(i))+ \
    (p*math.sin(v)/(1+e*math.cos(v)))*(-math.sin(raan)*math.sin(om)+math.cos(raan)*math.cos(om)*math.cos(i))
    
    z=(p*math.cos(v)/(1+e*math.cos(v)))*(math.sin(om)*math.sin(i))+(p*math.sin(v)/(1+e*math.cos(v)))*(math.cos(om)*math.sin(i))
    
    xdot=-math.sqrt(u/p)*math.sin(v)*(math.cos(raan)*math.cos(om)-math.sin(raan)*math.sin(om)*math.cos(i))+ \
    math.sqrt(u/p)*(e+math.cos(v))*(-math.cos(raan)*math.sin(om)-math.sin(raan)*math.cos(om)*math.cos(i))
    
    ydot=-math.sqrt(u/p)*math.sin(v)*(math.sin(raan)*math.cos(om)+math.cos(raan)*math.sin(om)*math.cos(i))+ \
    math.sqrt(u/p)*(e+math.cos(v))*(-math.sin(raan)*math.sin(om)+math.cos(raan)*math.cos(om)*math.cos(i))
    
    zdot=-math.sqrt(u/p)*math.sin(v)*(math.sin(om)*math.sin(i))+math.sqrt(u/p)*(e+math.cos(v))*(math.cos(om)*math.sin(i))
    
    cor=sci.array([x/r_nd,y/r_nd,z/r_nd],dtype="float64")
    vel=sci.array([xdot/v_nd,ydot/v_nd,zdot/v_nd],dtype="float64")
    
    return cor,vel


#masses
m1=1 #mass of sun
m2=0.000953 #mass of jupiter
m3=0.0002857 #mass of saturn

#initial position vectors
r1=[0,0,0] #m
r2=[1,0,0] #m
r3=[1.841,0,0] #m

#pv to arrays
r1=sci.array(r1,dtype="float64")
r2=sci.array(r2,dtype="float64")
r3=sci.array(r3,dtype="float64")

#initial velocities
v1=[0,0,0] 
v2=[0,1,0] 
v3=[0,0.7389,0] 

#vv to arrays
v1=sci.array(v1,dtype="float64")
v2=sci.array(v2,dtype="float64")
v3=sci.array(v3,dtype="float64")

#jupiter r and v
pj=7.7857e+11
ej=0.0489
ij=1.304
omj=273.367
raanj=100.55615
uj=1.327124e+20
vj=0

r2,v2=kep_car(pj,ej,ij,omj,raanj,uj,vj)

#saturn r and v
ps=1.43353e+12
es=0.0565
iss=2.485
oms=339.392
raans=113.665
us=1.327124e+20
vs=0

r3,v3=kep_car(ps,es,iss,oms,raans,us,vs)

#COM formula
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

#velocity of center of mass
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

#equations of motion
def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    
    r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)
    
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3
    
    r12_derivs=sci.concatenate((dr1bydt,dr2bydt))
    r_derivs=sci.concatenate((r12_derivs,dr3bydt))
    v12_derivs=sci.concatenate((dv1bydt,dv2bydt))
    v_derivs=sci.concatenate((v12_derivs,dv3bydt))
    derivs=sci.concatenate((r_derivs,v_derivs))

    return derivs    
    
#initial parameters
init_params=sci.array([r1,r2,r3,v1,v2,v3]) #array of initial param
init_params=init_params.flatten()
time_span=sci.linspace(0,50,10000)

#ode
import scipy.integrate
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]

#Find location of Alpha Centauri A w.r.t COM
r1com_sol=r1_sol-r_com
#Find location of Alpha Centauri B w.r.t COM
r2com_sol=r2_sol-r_com
#Find location of Third Star
r3com_sol=r3_sol-r_com


#plot figure
#define graph
plt.style.use('default')
fig=plt.figure(figsize=(10,10))


#create 3D axes
ax=fig.add_subplot(111,projection="3d")

#plot the orbits
ax.plot(r1com_sol[:,0],r1com_sol[:,1],r1com_sol[:,2],color="yellow")
ax.plot(r2com_sol[:,0],r2com_sol[:,1],r2com_sol[:,2],color="orange")
ax.plot(r3com_sol[:,0],r3com_sol[:,1],r3com_sol[:,2],color="blue")

#final position of stars
ax.scatter(r1com_sol[-1,0],r1com_sol[-1,1],r1com_sol[-1,2],color="yellow",\
           marker="o",s=100,label="Sun")
ax.scatter(r2com_sol[-1,0],r2com_sol[-1,1],r2com_sol[-1,2],color="orange",\
           marker="o",s=100,label="Jupiter")
ax.scatter(r3com_sol[-1,0],r3com_sol[-1,1],r3com_sol[-1,2],color="blue",\
           marker="o",s=100,label="Saturn")

ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Orbits in a three-body system",fontsize=14)
ax.legend(loc="upper left",fontsize=14)
