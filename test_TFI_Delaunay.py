import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import triangle as tr
import vedo

nu = 15
nv = 14
num_input = 2*(nu+nv)
x = np.linspace(0,1,nu)
y = np.linspace(0,1,nv)
u, v = np.meshgrid(x, y, indexing='ij')
segment_locations_all = np.stack((u,v), axis =2)
linear_map_matrix = np.zeros((nu*nv, num_input))
u = np.reshape(segment_locations_all[:,:,0],(nu*nv,1))
v = np.reshape(segment_locations_all[:,:,1],(nu*nv,1))
n = 0
for i in range(nu):
    for j in range(nv):
        linear_map_matrix[n,i] = 1-v[n]
        linear_map_matrix[n,i+nu] = v[n]
        linear_map_matrix[n,j+2*nu] = 1-u[n]
        linear_map_matrix[n,j+2*nu+nv] = u[n]
        linear_map_matrix[n,0] = linear_map_matrix[n,0] - (1-u[n])*(1-v[n])
        linear_map_matrix[n,2*nu-1] = linear_map_matrix[n,2*nu-1] - u[n]*v[n]
        linear_map_matrix[n,nu-1] = linear_map_matrix[n,nu-1] - u[n]*(1-v[n])
        linear_map_matrix[n,nu] = linear_map_matrix[n,nu] - v[n]*(1-u[n])
        n = n+1

nu = 15
nv = 14
nuu = nu-5
v0 = np.array([3, 3, 1])
v1 = np.array([20, 3, 0.3])
v2 = np.array([10, 10, -1])
v3 = np.array([3, 5, 0])
lu0 = np.linspace(v0,v1,num=nuu)
lu1 = np.linspace(v3,v2,num=nu)
lv0 = np.linspace(v0,v3,num=nv)
lv1 = np.linspace(v1,v2,num=nv)
xx = lu1[:,0]
yy = lu1[:,1]
zz = np.sin(x)
zz[0] = lu1[0,2]
zz[-1] = lu1[-1,2]
lu1 = np.dstack((xx,yy,zz))
lu1 = np.reshape(lu1,(nu,3))

lu_in = lu0
xu_in = lu_in[:,0]
yu_in = lu_in[:,1]
zu_in = lu_in[:,2]
u_in = np.linspace(0,1,num=nuu)
k = 2
tckx_in = interpolate.splrep(u_in,xu_in,k=k,s=0)
tcky_in = interpolate.splrep(u_in,yu_in,k=k,s=0)
tckz_in = interpolate.splrep(u_in,zu_in,k=k,s=0)
u_in = np.linspace(0,1,nu)
xxu_in = interpolate.splev(u_in,tckx_in).reshape((nu,1))
yyu_in = interpolate.splev(u_in,tcky_in).reshape((nu,1))
zzu_in = interpolate.splev(u_in,tckz_in).reshape((nu,1))
lu_in = np.concatenate((xxu_in,yyu_in,zzu_in),axis = 1)


indices_reduced = -1*np.ones((len(lu0),),dtype=np.int32)
print(np.shape(lu_in))
for i in range(len(lu0)):
    distance = 1e5
    for j in range(len(lu_in)):
        if np.linalg.norm(lu0[i]-lu_in[j])<distance and j not in indices_reduced:
            indices_reduced[i] =j
            distance = np.linalg.norm(lu0[i]-lu_in[j])
print(indices_reduced)
points_lu0 = vedo.Points(lu0, r= 30, c='yellow', alpha =0.3).legend('number 10')
points_lu1 = vedo.Points(lu1, r= 20, c='green').legend('number 15')
points_lu_in = vedo.Points(lu_in, r= 25, c='blue', alpha =0.5).legend('number 15')
points_lu_in_reduced = []
for kk in indices_reduced:
    points_lu_in_reduced.append(vedo.Points(lu_in[kk,:].reshape((1,3)), r= 15, c='black'))
points_lv0 = vedo.Points(lv0, r= 15, c='red')
points_lv1 = vedo.Points(lv1, r= 15, c='red')
vedo_plot0 = vedo.Plotter()
vedo_plot0.show('Boundaries: yellow--10; black--10; green--15; blue--15', points_lu0, points_lv0, points_lv1, points_lu1, points_lu_in,points_lu_in_reduced, axes=1, interactive = False)


#parau1 = np.concatenate((u1.reshape(len(u1),1),np.zeros(len(u1)).reshape(len(u1),1)),axis=1)

#indices_reduced = list(range(5)) + list(range(6,len(x)))
x_in = x[indices_reduced]
print(np.shape(x))
print(np.shape(x_in))
parau0 = np.stack((x,np.zeros((nu,))),axis = -1)
parau1 = np.stack((x,np.ones((nu,))),axis=-1)
parav0 = np.stack((np.zeros((nv,)),y),axis=-1)
parav1 = np.stack((np.ones((nv,)),y),axis=-1)
boundaries = np.concatenate((parau0,parau1,parav0,parav1))
print(np.shape(boundaries))
vertices_uv = np.matmul(linear_map_matrix,boundaries)
print('vertices_uv',np.shape(vertices_uv))

diff = np.setdiff1d(x, x_in)
vertices_uv_reduced = np.zeros((np.shape(vertices_uv)[0] - len(diff),2))

boundaries_xyz = np.concatenate((lu_in,lu1,lv0,lv1))
vertices_xyz = np.matmul(linear_map_matrix,boundaries_xyz)
print('vertices_xyz',np.shape(vertices_xyz))
vertices_xyz_reduced = np.zeros((np.shape(vertices_xyz)[0] - len(diff),3))

j = 0
for i in range(np.shape(vertices_uv)[0]):
    if vertices_uv[i][0] in diff and vertices_uv[i][1] == 0:
        pass
    else:
        vertices_uv_reduced[j,:] = vertices_uv[i,:]
        vertices_xyz_reduced[j,:] = vertices_xyz[i,:]
        j += 1
print('vertices_uv_reduced',np.shape(vertices_uv_reduced))
print('vertices_xyz_reduced',np.shape(vertices_xyz_reduced))

edges = np.empty((0,2), dtype=np.int32)
count = 0
for l in [x_in,x,y,y]:
    for k in range(np.shape(l)[0]-1): 
        #print(np.shape(l)[0]-1)
        if count == 0:
            #print(x_in[k], x_in[k+1])
            start = np.where((vertices_uv_reduced == (x_in[k], 0)).all(axis=1))[0]
            end = np.where((vertices_uv_reduced == (x_in[k+1], 0)).all(axis=1))[0]
            edges = np.concatenate((edges, np.array(([start,end])).reshape(1,2))) 
        elif count == 1: 
            start = np.where((vertices_uv_reduced == (x[k], 1)).all(axis=1))[0]
            end = np.where((vertices_uv_reduced == (x[k+1], 1)).all(axis=1))[0]
            edges = np.concatenate((edges, np.array(([start,end])).reshape(1,2)))
        elif count == 2:
            start = np.where((vertices_uv_reduced == (0, y[k])).all(axis=1))[0]
            end = np.where((vertices_uv_reduced == (0, y[k+1])).all(axis=1))[0]
            edges = np.concatenate((edges, np.array(([start,end])).reshape(1,2)))
        elif count == 3:
            start = np.where((vertices_uv_reduced == (1, y[k])).all(axis=1))[0]
            end = np.where((vertices_uv_reduced == (1, y[k+1])).all(axis=1))[0]
            edges = np.concatenate((edges, np.array(([start,end])).reshape(1,2)))            
    count += 1    
print(np.shape(edges))
print(vertices_uv_reduced.shape,edges.shape)
exit()
A = dict(vertices=vertices_uv_reduced, segments=edges)#
B = tr.triangulate(A,'pc')
tr.compare(plt, A, B)
plt.show(block=False)
# exit()
# print(edges)
length = len(B['triangles'].tolist())
edges_uv_plot = []
edges_xyz_plot = []
for i in range(np.shape(edges)[0]):
    edges_uv_plot.append(vedo.Points(vertices_uv_reduced[edges[i,0],:].reshape((1,2)), r= 20, c='red'))
    edges_uv_plot.append(vedo.Points(vertices_uv_reduced[edges[i,1],:].reshape((1,2)), r= 20, c='red'))
    edges_xyz_plot.append(vedo.Points(vertices_xyz_reduced[edges[i,0],:].reshape((1,3)), r= 20, c='red'))
    edges_xyz_plot.append(vedo.Points(vertices_xyz_reduced[edges[i,1],:].reshape((1,3)), r= 20, c='red'))
mesh = vedo.Mesh([vertices_uv_reduced, np.array(B['triangles'].tolist()).reshape((length,3))])
mesh.backColor().lineColor('green').lineWidth(3)
points_uv = vedo.Points(vertices_uv_reduced, r= 15, c='blue')
vedo_plot1 = vedo.Plotter()
vedo_plot1.show('2D mesh', mesh,edges_uv_plot, points_uv, axes=1, interactive = False)


mesh_xyz = vedo.Mesh([vertices_xyz_reduced, np.array(B['triangles'].tolist()).reshape((length,3))])
mesh_xyz.backColor().lineColor('green').lineWidth(3)
points_xyz = vedo.Points(vertices_xyz_reduced, r= 15, c='blue')
points_lu_in_reduced = []
for kk in indices_reduced:
    points_lu_in_reduced.append(vedo.Points(lu_in[kk,:].reshape((1,3)), r= 25, c='black'))
vedo_plot2 = vedo.Plotter()
vedo_plot2.show('3D mesh', mesh_xyz,points_xyz,edges_xyz_plot,points_lu_in_reduced, axes=1, interactive = True)

import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
nx, ny = (5, 4)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 20, ny)
xv, yv = np.meshgrid(x, y, sparse = True)
verts = np.zeros((nx*ny,2))
count = 0
for xx in xv.reshape((nx,1)):
    for yy in yv.reshape((ny,1)):
        verts[count,0] = xx
        verts[count,1] = yy
        count += 1
verts = np.delete(verts,9,0)
verts = np.delete(verts,9,0)
edge =  np.linspace(np.array([5,0]), np.array([5,20]),5)   
edge = edge[1:-1,:]  
print(edge)
edge2 =  np.linspace(np.array([0,12]), np.array([10,8]),5) 
vertices1 = np.concatenate((verts,edge,edge2))
print(np.shape(vertices1)) 
edge = np.array(([[0,3],[17,3],[17,14],[0,14],[2,6],[6,12],[12,16]]), dtype = np.int32)#,[21,22]
print(np.shape(edge)) 
A = dict(vertices=vertices1, segments=edge)#
B = tr.triangulate(A,'pc')
#print(B)
tr.compare(plt, A, B)
A = dict(vertices=vertices1)#, segments=edge
B = tr.triangulate(A)
tr.compare(plt, A, B)
plt.show()