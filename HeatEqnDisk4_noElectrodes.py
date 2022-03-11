# USE THIS ONE!!!

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import copy
import pickle
import datetime

def heaviside(x):
    return (x > 0).astype(float)

T_amb = 293.0 # ambient temperature in C

tol = 1e-5 # tolerance of simulation
k = 10e-3 # # thermal conductivity of quartz in 10e-3 W/(mm K)

N_r = 1000 # number of cells in r-direction
r_min = 0. # mm
r_max = 9.525 / 2. # mm
dr = (r_max - r_min) / N_r # step size in r-direction

dphi = 2 * np.pi

N_z = 100 # number of cells in z-direction
z_min = 0. 
z_max = 0.1 #mm 0.127

# Substrate node r coords
r = np.arange(dr/2., r_max, dr)
Rmax = np.max(r)

# Substrate node z coords
N_z = int(np.floor((z_max - z_min) / dr))
z = np.linspace(z_min, z_max, N_z)
dz = z[1] - z[0]
t = z[-1] - z[0]
#print('dz = {}'.format(dz))

# Make Computational mesh from (r,z) coords
R, Z = np.meshgrid(r,z)
print(R.shape)

# Setup internal heat generation and nodal conductances
Q_laser = 24e-3 # W
r_extent_top = 0.5 # mm
q_Ai = Q_laser / (np.pi * r_extent_top**2.)
q = np.zeros(np.shape(R))
C_mp = np.zeros(np.shape(R))
C_mm = np.zeros(np.shape(R))
C_kp = np.zeros(np.shape(R))
C_km = np.zeros(np.shape(R))
substrate_coeff_R = k * dphi * dz / dr
substrate_coeff_Z = k * dphi * dr / dz
print('r coeff ',substrate_coeff_R)
print('z coeff ',substrate_coeff_Z)

for j in range(len(r)):
    for i in range(len(z)):
        # Internal Heat generation
        if i == len(z)-1:
            q[i,j] = q_Ai * heaviside(r_extent_top - R[i,j]) * R[i,j] * dr * dphi
            
        if R[i,j] == Rmax:
            # edge zone is only half way to other zone
            C_mp[i,j] = 2 * (R[i,j] + dr/4) * substrate_coeff_R
        else:
            C_mp[i,j] = (R[i,j] + dr/2) * substrate_coeff_R
        # first zone inward radial conductance cancels, since we start at dr/2
        C_mm[i,j] = (R[i,j] - dr/2) * substrate_coeff_R
        C_kp[i,j] = R[i,j] * substrate_coeff_Z
        C_km[i,j] = R[i,j] * substrate_coeff_Z


weight_sum = C_mp + C_mm + C_kp + C_km

def heatcontour(R, Z, data, nodes=False):
    levels = MaxNLocator(nbins=15).tick_values(data.min(), data.max())
    cmap = plt.get_cmap('autumn')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax0 = plt.subplots(nrows=1)

    cf = ax0.contourf(R, Z, data, levels=levels, cmap=cmap)
    if nodes:
        nodes = []
        for ri in R[0,:]:
            for zj in Z[:,0]:
                nodes.append([ri,zj])
        nodes = np.array(nodes)
        plt.plot(nodes[:,0], nodes[:,1],'w+')
    fig.colorbar(cf, ax=ax0)
    plt.show()

def plot_convergence(xdata, ydata1, ydata2, title):
    plt.loglog(xdata, np.absolute(np.array(ydata1)/ydata1[0]),'r.')
    plt.loglog(xdata, np.absolute(np.array(ydata2)/ydata2[0]),'b.')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(r"$|T/T_{start}|$")
    plt.show()

R_shape = np.shape(R)

# Constant that accounts for area changes in sectors when considering heat flow through top/bottom faces
C_n = (2 * dz)/(k * r * dphi * dr)
weight_sum_bot = C_mp[0,:] + C_mm[0,:] + C_kp[0,:]
weight_sum_top = C_mp[-1,:] + C_mm[-1,:] + C_km[-1,:]

# Correction factor as determined by binary search in "HeatEqnDisk4_noElectrodes_binarySearch.py
mid = 0.9554177913814783
# low, mid, hi bracketing from binary search
bracket = [0.9554177820682526, 0.9554177913814783, 0.9554178006947041]

# Define initial and boundary conditions

Tmid = (Q_laser / (2*np.pi*k*t)) * (np.log(R[0,-1]/r_extent_top) + 0.5 - t**2/r_extent_top**2 + Z**2/r_extent_top**2)
T = np.zeros(R.shape)

for i in range(R_shape[0]):
    for j in range(R_shape[1]-1):
        if R[i,j] <= r_extent_top:
            T[i,j] = Tmid[i,j] - Tmid[i,j] * Q_laser * R[i,j] / (4 * np.pi * r_extent_top * k * t * Tmid[-1,0])
        else:
            T[i,j] = (Tmid[i,j]/Tmid[-1,0]) * Q_laser * np.log(r_max / R[i,j]) / (2 * np.pi * k * t)

T = mid*T + T_amb

# Neumann BCs in z
# Antiquated: Default 
q_laser = 0#Q_laser / (np.pi * r_extent_top**2) # W/mm^2
qrTop = -q_laser * heaviside(r_extent_top - r)
if qrTop is not None:
    TNeuTop = T[-2,:] #- C_n * r * qrTop

r_extent_bot = 1.
qrBot = 0 *  heaviside(r_extent_bot - r) #np.exp(-r**2 / 2. / 1.) *
if qrBot is not None:
    TNeuBottom = T[1,:] #+ C_n * r * qrBot
# End antiquated tag

print('MCL')
iter_no = [0]
T_top = [T[0,0]-T_amb]
T_bottom = [T[-1,0]-T_amb]
eps = 1.
iters = 0
end_iter = 400000
#--- MAIN COMPUTATIONAL LOOP --- GAUSS-SEIDEL ---
#while eps > tol:
while iters < end_iter:
    Tcopy = copy.copy(T)
    for i in range(R_shape[0]): # loop over z
        for j in range(R_shape[1] - 1): # loop over r. last zone is BC
            if i == 0 and TNeuBottom is not None:
                # lower z Neumann BC
                Tcopy[i,j] = (q[i,j] + (C_mp[i,j]*T[i,j+1] + C_mm[i,j]*T[i,j-1] + C_kp[i,j]*T[i+1,j] )) / weight_sum_bot[j]
            elif i == R_shape[0]-1 and TNeuTop is not None:
                # upper z Neumann BC
                Tcopy[i,j] = (q[i,j] + (C_mp[i,j]*T[i,j+1] + C_mm[i,j]*T[i,j-1] + C_km[i,j]*T[i-1,j])) / weight_sum_top[j]
            else:
                Tcopy[i,j] = (q[i,j] + (C_mp[i,j]*T[i,j+1] + C_mm[i,j]*T[i,j-1] + C_kp[i,j]*T[i+1,j] + C_km[i,j]*T[i-1,j])) / weight_sum[i,j]
                    
    # Update Ghost nodes, if needed
    if TNeuBottom is not None:
        TNeuBottom = Tcopy[1,:] #+ C_n * r * qrBot
    if TNeuTop is not None:
        TNeuTop = Tcopy[-2,:] #- C_n* r * qrTop
    
    eps = Tcopy[-1,0] - T[-1,0] #np.max(np.absolute(T-Tcopy))
    iters += 1
    
    if iters % 500 == 0:
        print('Top node changing by {} each iteration after {} iterations...'.format(eps,iters))
        #rint(Tcopy[-1,0])
        iter_no.append(iters)
        T_bottom.append(T[0,0]-T_amb)
        T_top.append(T[-1,0]-T_amb)

    '''
    if iters % 5000 == 0:
        heatcontour(R, Z, T)
    '''    

    '''
    if iters == end_iter - 1:
        heatcontour(R, Z, T)
    '''
        
    T = Tcopy
#'''
print('Finished after {} iterations...'.format(iters))

plot_convergence(iter_no, T_top, T_bottom, "Top and Bottom Temperatures vs. Iteration")

# Plotting
def plot_heatmap():
    levels = MaxNLocator(nbins=15).tick_values(1000*(T.min()-T_amb), 1000*(T.max()-T_amb))
    cmap = plt.get_cmap('jet')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax0 = plt.subplots(nrows=1)

    cf = ax0.contourf(R, Z, 1000*(T-T_amb), levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)

    # contour plot is good for printing.
    cs = plt.contour(R, Z, 1000*(T-T_amb), 15, colors='white', linestyles='solid')
    plt.clabel(cs, fmt='%.2f', inline=True)
    plt.xlabel("R [mm]")
    plt.ylabel("Z [mm]")

    plt.show()

plot_heatmap()

def save_data():
    now = datetime.datetime.now()
    currtime = now.strftime("%m%d%y_%H%M")

    print(" Saving data to disk...")
    data = {'T': T,
            'R': R,
            'Z': Z,
            'q': q,
            }

    f = open('HeatEqnDisk4_noElectrodes_'+currtime+'.pckl', 'wb')
    pickle.dump(data, f)
    f.close()

save_data()
