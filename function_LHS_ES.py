#Latin Hypercube sampling https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS
from scipy import interpolate
import vedo 
import triangle as tr

num_u0 = 4
num_u1 = 15
num_v0 = 7
num_v1 = 15
num_v = 6

point_00 = np.array([3, 3, 1])
point_01 = np.array([5, 3, 0.3])
point_10 = np.array([3, 5, 0])
point_11 = np.array([10, 7, -1])

l_u0 = np.linspace(point_00, point_01, num=num_u0)
l_u1 = np.linspace(point_10, point_11, num=num_u1)
l_v0 = np.linspace(point_00, point_10, num=num_v)
l_v1 = np.linspace(point_01, point_11, num=num_v)
x_u1 = l_u1[:,0]
y_u1 = l_u1[:,1]
z_u1 = np.sin(x_u1)
z_u1[0] = l_u1[0,2]
z_u1[-1] = l_u1[-1,2]
l_u1 = np.dstack((x_u1,y_u1,z_u1))
l_u1 = np.reshape(l_u1,(num_u1,3))

vedo_u0 = vedo.Points(l_u0, r= 15, alpha = 0.5, c='yellow')
vedo_u1 = vedo.Points(l_u1, r= 15, alpha = 0.5, c='blue')
vedo_v0 = vedo.Points(l_v0, r= 13, alpha = 0.5, c='red')
vedo_v1 = vedo.Points(l_v1, r= 13, alpha = 0.5, c='red')
# plot1 = vedo.Plotter()
# plot1.show('Boundary', vedo_u0,  vedo_v0, vedo_u1, vedo_v1, axes=1, viewup="z", interactive=False) 

length_u0 = np.linalg.norm(point_00 - point_01)
length_u1 = np.linalg.norm(point_01 - point_11)
if num_u0 > num_u1:
    num_u_max, num_u_min = num_u0,  num_u1
    u_max_length, u_min_length = length_u0, length_u1 
else: 
    num_u_max, num_u_min = num_u1,  num_u0
    u_max_length, u_min_length = length_u1, length_u0
print('num_u_max',num_u_max,'num_u_min',num_u_min)
print('length_u_max',u_max_length,'length_u_min', u_min_length)

num_uu = num_u_max
num_vv = num_v
uu = np.linspace(0, 1, num=num_uu)
vv = np.linspace(0, 1, num=num_vv)
u_lower = uu[0]
u_upper = uu[-1]
num_sampling = np.round(np.linspace(num_u_min, num_u_max+0.49, num_vv-1)).astype(int)
print('num_sampling',num_sampling)
vedo_internal_pts = []
uv = np.empty((0,2)) 
for i in range(num_vv-1):
    if num_u_max == num_u0:
        v_lower = vv[i]
        v_upper = vv[i+1]
    else:
        v_lower = vv[-i-1]
        v_upper = vv[-i-2]
    limits = np.array([[0, 1], [v_lower, v_upper]])
    sampling = LHS(xlimits = limits)
    x = sampling(num_sampling[-i-1])
    #print(x.shape)
    uv = np.concatenate((uv,x))
    vedo_internal_pts.append(vedo.Points(x, r= 13, c=i))

print('uv', uv.shape)
num_internal = uv.shape[0]
u0 = np.vstack((np.linspace(0, 1, num=num_u0),np.linspace(0, 0, num=num_u0))).T
u1 = np.vstack((np.linspace(0, 1, num=num_u1),np.linspace(1, 1, num=num_u1))).T
v0 = np.vstack((np.linspace(0, 0, num=num_v),np.linspace(0, 1, num=num_v))).T
v1 = np.vstack((np.linspace(1, 1, num=num_v),np.linspace(0, 1, num=num_v))).T
ve_u0 = vedo.Points(u0, r= 15, alpha = 0.5, c='black')
ve_u1 = vedo.Points(u1, r= 15, alpha = 0.5, c='black')
ve_v0 = vedo.Points(v0, r= 15, alpha = 0.5, c='black')
ve_v1 = vedo.Points(v1, r= 15, alpha = 0.5, c='black')

np.savetxt('test_uv.txt', uv) 
uv = np.loadtxt('test_uv.txt') 

plot2 = vedo.Plotter()
axes_opts = dict(
    xtitle='u', # latex-style syntax
    ytitle='v')
plot2.show('LHS_uv_hu',vedo_internal_pts, ve_u0,  ve_v0, ve_u1, ve_v1, axes=axes_opts, interactive = False) 

k = 2
u0 = np.linspace(0,1,num=num_u0)
tckx0 = interpolate.splrep(u0,l_u0[:,0],k=k,s=0)
tcky0 = interpolate.splrep(u0,l_u0[:,1],k=k,s=0)
tckz0 = interpolate.splrep(u0,l_u0[:,2],k=k,s=0)
u1 = np.linspace(0,1,num=num_u1)
tckx2 = interpolate.splrep(u1,l_u1[:,0],k=k,s=0)
tcky2 = interpolate.splrep(u1,l_u1[:,1],k=k,s=0)
tckz2 = interpolate.splrep(u1,l_u1[:,2],k=k,s=0)
v0 = np.linspace(0,1,num=num_v)
tckx1 = interpolate.splrep(v0,l_v0[:,0],k=k,s=0)
tcky1 = interpolate.splrep(v0,l_v0[:,1],k=k,s=0)
tckz1 = interpolate.splrep(v0,l_v0[:,2],k=k,s=0)
v1 = np.linspace(0,1,num=num_v)
tckx3 = interpolate.splrep(v1,l_v1[:,0],k=k,s=0)
tcky3 = interpolate.splrep(v1,l_v1[:,1],k=k,s=0)
tckz3 = interpolate.splrep(v1,l_v1[:,2],k=k,s=0)
u = uv[:,0]
v = uv[:,1]
vertcoord = np.zeros((uv.shape[0],3))
vertcoord[:,0] = (np.multiply((1-v),interpolate.splev(u,tckx0)) + np.multiply(v,interpolate.splev(u,tckx2))+np.multiply((1-u),interpolate.splev(v,tckx1)) + np.multiply(u,interpolate.splev(v,tckx3))\
    -(np.multiply((1-u),(1-v))*point_00[0] + np.multiply(u,v)*point_11[0] + np.multiply(u,(1-v))*point_01[0] + np.multiply((1-u),v)*point_10[0])).flatten(order='C')
vertcoord[:,1] = (np.multiply((1-v),interpolate.splev(u,tcky0)) + np.multiply(v,interpolate.splev(u,tcky2)) + np.multiply((1-u),interpolate.splev(v,tcky1)) + np.multiply(u,interpolate.splev(v,tcky3))\
    -(np.multiply((1-u),(1-v))*point_00[1] + np.multiply(u,v)*point_11[1] + np.multiply(u,(1-v))*point_01[1] + np.multiply((1-u),v)*point_10[1])).flatten(order='C')
vertcoord[:,2] = (np.multiply((1-v),interpolate.splev(u,tckz0)) + np.multiply(v,interpolate.splev(u,tckz2))+ np.multiply((1-u),interpolate.splev(v,tckz1)) + np.multiply(u,interpolate.splev(v,tckz3))\
    -(np.multiply((1-u),(1-v))*point_00[2] + np.multiply(u,v)*point_11[2] + np.multiply(u,(1-v))*point_01[2] + np.multiply((1-u),v)*point_10[2] )).flatten(order='C')

points1 = vedo.Points(vertcoord, r = 10, c='green')
plot3 = vedo.Plotter()
plot3.show('LHS_TFI_3D_hu', vedo_u0,  vedo_v0, vedo_u1, vedo_v1, points1, axes=1, interactive = True)
print('uv',uv.shape)
#print(np.sum(num_sampling)+num_u0+num_u1+num_v)
exit()
fig, ax = plt.subplots()

N = uv.shape[0]
ndim = 2
masses = np.ones(N)
charges = np.ones((N,))#array([1, 1, 1, 1, 1]) * 2
print('charges',np.shape(charges))
loc_arr = vertcoord#uv
print('loc_arr',np.shape(loc_arr))
speed_arr = np.zeros((N, ndim))
charge_matrix = -1 * np.outer(charges, charges) # compute charge matrix, ie c1 * c2
print('charge_matrix',charge_matrix.shape)
time = np.linspace(0, 0.03)
dt = np.ediff1d(time).mean()
print('dt:',dt)#0.00061224
#print(np.ediff1d(time)) all 0.00061224 

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
    marker='.'
    # plotting
    if not i:
        color = 'k'
        zorder = 3
        ms = 3
        vedo_internal_pts1 = vedo.Points(loc_arr, r= 13, c=3,alpha = 0.5)
        marker='.'
        ax.plot(loc_arr[:,0], loc_arr[:,1], '*', color=color, ms=ms, zorder=zorder, label = 'Start')
        # for i, pt in enumerate(loc_arr):
        #     ax.text(*pt + 0.1, s='{}q {}m'.format(charges[i], masses[i]))
    elif i == len(time)-1:
        color = 'b'
        zroder = 3
        ms = 3
        vedo_internal_pts2 = vedo.Points(loc_arr, r= 13, c=1)
        marker='.'
        loc_final = loc_arr
        ax.plot(loc_arr[:,0], loc_arr[:,1], '^', color=color, ms=ms, zorder=zorder, label = 'End')
    else:
        color = 'r'
        zorder = 1
        ms = 1
        marker='.'
        ax.plot(loc_arr[:,0], loc_arr[:,1], '.', color=color, ms=ms, zorder=zorder)

ax.set_aspect('equal')
ax.set_title('Rutherford scattering')
plt.legend()

for i in range(2):
    norm_constant =  max(loc_final[:,i]) - min(loc_final[:,i]) 
    min_value = min(loc_final[:,i])
    max_value = max(loc_final[:,i])
    loc_final[:,i] = (0.9*(loc_final[:,i] - min_value))/ norm_constant + 0.05

vedo_internal_pts3 = vedo.Points(loc_final, r= 13, c=2)
plot2 = vedo.Plotter()
axes_opts = dict(
    xtitle='u', # latex-style syntax
    ytitle='v')
plot2.show('ES_uv',vedo_internal_pts3, ve_u0,  ve_v0, ve_u1, ve_v1, axes=axes_opts, interactive = False)

uv = loc_final
uv = np.concatenate((uv,u0))
uv = np.concatenate((uv,u1))
uv = np.concatenate((uv,v0))
uv = np.concatenate((uv,v1))

edge = np.zeros((num_u0+num_u1+num_v+num_v-4,2), dtype = np.int32)
for i in range(num_u0+num_u1+num_v+num_v-4):
    edge[i,0] = i + num_internal
    edge[i,1] = i + num_internal + 1
    if i == num_u0+num_u1+num_v+num_v-5:
        edge[i,0] = i + num_internal
        edge[i,1] = num_internal
print(uv.shape, edge.shape)
edge = np.array(([[49,50],[50,51]]), dtype = np.int32)
print(edge)
A = dict(vertices=uv)#, segments=edge)
B = tr.triangulate(A,'pc')
#print(B)
tr.compare(plt, A, B)
plt.show(block=False)


