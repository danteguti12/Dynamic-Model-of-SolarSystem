#three star system

import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#Universal constant
G=6.67408e-11 #Nm2/kg2

#physical quantities
m_nd=1.989e+30 #mass of the sun kg
r_nd=5.326e+12 #distance between stars
v_nd=30000 #velocity of earth with respect to the sun
t_nd=79.91*365*24*3600*0.51 #orbital period of alpha centauri

#net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#masses
m1=1.1 #alpha centauri a
m2=0.907 #alpha centauri b
m3=1.0 #Third Star

#initial position vectors
r1=[-0.5,0,0] #m
r2=[0.5,0,0] #m
r3=[0,1,0] #m

#pv to arrays
r1=sci.array(r1,dtype="float64")
r2=sci.array(r2,dtype="float64")
r3=sci.array(r3,dtype="float64")

#initial velocities
v1=[0.01,0.01,0] #m/s
v2=[-0.05,0,-0.1] #m/s
v3=[0,-0.01,0] #m/s

#vv to arrays
v1=sci.array(v1,dtype="float64")
v2=sci.array(v2,dtype="float64")
v3=sci.array(v3,dtype="float64")

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
time_span=sci.linspace(0,20,500)

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
fig=plt.figure(figsize=(15,15))

#create 3D axes
ax=fig.add_subplot(111,projection="3d")

#plot the orbits
ax.plot(r1com_sol[:,0],r1com_sol[:,1],r1com_sol[:,2],color="darkblue")
ax.plot(r2com_sol[:,0],r2com_sol[:,1],r2com_sol[:,2],color="tab:red")
ax.plot(r3com_sol[:,0],r3com_sol[:,1],r3com_sol[:,2],color="green")

#final position of stars
ax.scatter(r1com_sol[-1,0],r1com_sol[-1,1],r1com_sol[-1,2],color="darkblue",\
           marker="o",s=100,label="Alpa Centauri A")
ax.scatter(r2com_sol[-1,0],r2com_sol[-1,1],r2com_sol[-1,2],color="tab:red",\
           marker="o",s=100,label="Alpa Centauri B")
ax.scatter(r2com_sol[-1,0],r2com_sol[-1,1],r2com_sol[-1,2],color="green",\
           marker="o",s=100,label="Third Star")

ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Orbits in a three-body system",fontsize=14)
ax.legend(loc="upper left",fontsize=14)

