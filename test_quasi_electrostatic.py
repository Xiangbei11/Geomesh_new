# # point_ charge.py - Iterative solution of 2-D PDE, ele
# import matplotlib
# import numpy as np
# import matplotlib. pyplot as plt
# import vedo
# # Set dimensions of the probl em
# L=1.0
# N=21
# ds = L/N
# # Define arrays used for plotting
# x = np.linspace(0 ,L,N)
# y = np.copy (x)
# X, Y = np. meshgrid(x,y)
# # Make the charge density matrix
# rho0=1.0
# rho = np.zeros((N,N))
# rho [ int ( round(N/2.0)) , int (round(N/2.0))] = rho0
# # Make the initial guess for solution matrix
# V = np.zeros((N,N))
# # Solver
# iterations = 0
# eps = 1e-8
# # Convergence threshold
# error = 1e4 # Large dummy error
# while iterations < 1e4 and error > eps :
#     V_temp = np.copy (V)
#     error = 0 # we make this accumulate in the loop
#     for j in range(2,N-1) :
#         for i in range(2,N-1) :
#             V[i,j] = 0.25*(V_temp[i+1,j] + V_temp[i-1,j] +V_temp[i,j-1] + V_temp[i,j+1] + rho[i, j]*ds**2)
#             error += abs(V[i,j]-V_temp [i , j])
#     iterations += 1
#     error /= float (N)
# print( "iterations =" ,iterations )
# print(np.shape(np.stack((x,y))))
# add = V.flatten()*1e4
# vedo_internal_pts1 = vedo.Points(np.stack((X.flatten()+add,Y.flatten()+add)).T, r= 13, c=i)
# vedo_internal_pts2 = vedo.Points(np.stack((X.flatten(),Y.flatten())).T, r= 13, c=3)
# print(np.shape(V.flatten()*1e4))
# #vedo_internal_pts2 = vedo.Points(V, r= 13, c=i)
# plot2 = vedo.Plotter()
# axes_opts = dict(
#     xtitle='u', # latex-style syntax
#     ytitle='v')
# plot2.show('LHS_uv',vedo_internal_pts1,vedo_internal_pts2, axes=axes_opts, interactive = True) 
# # # Plotting
# # matplotlib.rcParams['xtick.direction'] = 'out'
# # matplotlib.rcParams['ytick.direction'] = 'out'
# # CS = plt.contour(X,Y,V ,30) # Make a contour plot
# # plt. clabel(CS, inline=1,
# # fontsize=10)
# # plt. title('PDE solution of a point charge' )
# # CB = plt. colorbar(CS, shrink=0.8, extend=' both')
# # plt. show()

from matplotlib import pyplot as plt
import numpy as np
import vedo
uv = np.loadtxt('test_uv.txt') 

fig, ax = plt.subplots()

N = 49
ndim = 2
masses = np.ones(N)
charges = np.ones((N,))#array([1, 1, 1, 1, 1]) * 2
print(np.shape(charges))
# loc_arr = np.random.rand(N, ndim)
loc_arr = uv#np.array(((0.1,0.7), (0.3,0.5), (0.4,0.1), (0.7,0.2), (0.9,0.8)), dtype=float)
print(np.shape(loc_arr))
#exit()
speed_arr = np.zeros((N, ndim))

# compute charge matrix, ie c1 * c2
charge_matrix = -1 * np.outer(charges, charges)

time = np.linspace(0, 0.02)
dt = np.ediff1d(time).mean()
np.seterr(invalid='ignore')
for i, t in enumerate(time):
    # get (dx, dy) for every point
    delta = (loc_arr.T[..., np.newaxis] - loc_arr.T[:, np.newaxis]).T
    # calculate Euclidean distance
    distances = np.linalg.norm(delta, axis=-1)
    # and normalised unit vector
    unit_vector = np.divide(delta.T, distances).T#(delta.T / distances).T
    unit_vector[np.isnan(unit_vector)] = 0 # replace NaN values with 0
    # calculate force
    np.seterr(divide='ignore',invalid='ignore')
    force = np.divide(charge_matrix, distances**2)#charge_matrix / distances**2 # norm gives length of delta vector
    force[np.isinf(force)] = 0 # NaN forces are 0

    # calculate acceleration in all dimensions
    acc = (unit_vector.T * force / masses).T.sum(axis=1)
    # v = a * dt
    speed_arr += acc * dt

    # increment position, xyz = v * dt
    loc_arr += speed_arr * dt 

    # plotting
    if not i:
        color = 'k'
        zorder = 3
        ms = 3
        vedo_internal_pts1 = vedo.Points(loc_arr, r= 13, c=3,alpha = 0.5)
        # for i, pt in enumerate(loc_arr):
        #     ax.text(*pt + 0.1, s='{}q {}m'.format(charges[i], masses[i]))
    elif i == len(time)-1:
        color = 'b'
        zroder = 3
        ms = 3
        vedo_internal_pts2 = vedo.Points(loc_arr, r= 13, c=1)
        loc_final = loc_arr
    else:
        color = 'r'
        zorder = 1
        ms = 1
    ax.plot(loc_arr[:,0], loc_arr[:,1], '.', color=color, ms=ms, zorder=zorder)

ax.set_aspect('equal')
#plt.show()
for i in range(2):
    norm_constant =  max(loc_final[:,i]) - min(loc_final[:,i]) 
    min_value     =  min(loc_final[:,i])
    loc_final[:,i] = (loc_final[:,i] - min_value) / norm_constant 

vedo_internal_pts3 = vedo.Points(loc_final, r= 13, c=2)
plot2 = vedo.Plotter()
axes_opts = dict(
    xtitle='u', # latex-style syntax
    ytitle='v')
plot2.show('LHS_uv',vedo_internal_pts1,vedo_internal_pts3, axes=axes_opts, interactive = True)