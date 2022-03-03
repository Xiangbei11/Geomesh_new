import numpy as np
import vtk
import vedo
from smt.sampling_methods import LHS

import pymeshopt
from pyoctree import pyoctree as ot
from member_new import Member

from scipy import interpolate

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# from options_dictionary import OptionsDictionary
# from meshopt import meshopt

# from scipy.spatial import Delaunay
# import math
# import stl
# import meshio

# import time


class Geomesh(object):
    def __init__(self, path):
        ''' 
        ----------
        path : str
            Location of the stl/vtk file for the initial mesh.
        ----------
        self.members : list of instances of member
        self.num_members : number of members
        self.quadlist : np.ndarray (int32) of indices of the quads (num_quad,4)
        self.projected : list of all the projected points

        mem_name : dictionary of member and names
        projection : list of all the projection points         
        projection_dir : list of all the projection directions        
        nonintlist : list (string) of all the name of the members between which the intersections are not expected (num,2) 
        '''
        self.path = path
        self.members = []
        self.num_members = 0
        self.quadlist = np.empty((0,4), dtype=np.int32)
        self.projected = [] 
        
        ###
        # self.mem_name = {}        
        # self.projection = []
        # self.projected 
        # self.projection_point = []
        # self.projection_dir = []
        # self.nonintlist = []
         
    def compute_topology(self, order = [0,1,2], plot = False):
        ''' Get array of idinates of vertices and polygon connectivities.
        ----------
        order : np.array
            A numpy array defining the expected ordering of the idinates of the vertices.
            For example, if order = [0,2,1], the y and z idinates of the topology will switch.
        plot : bool
            If True, allow vedo plotting of OML triangular mesh.
        '''
        if self.path[-4:].lower() == '.stl':
            reader = vtk.vtkSTLReader()
            reader.MergingOn()
        elif self.path[-4:].lower() == '.vtk':
            reader = vtk.vtkUnstructuredGridReader()
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
        else:
            print("Please input an iges file or a stp file.")
        
        reader.SetFileName(self.path)
        reader.Update()
        data = reader.GetOutput()

        numPoints = data.GetNumberOfPoints()
        pointCoords = np.zeros((numPoints, 3))
        for i in range(numPoints):
            pointCoords[i,:] = data.GetPoint(i)

        numPolys = data.GetNumberOfCells()     
        connectivity = np.zeros((numPolys, 3), dtype=np.int32)
        for i in range(numPolys):
            atri = data.GetCell(i)
            ids = atri.GetPointIds()
            for j in range(3):
                connectivity[i,j] = ids.GetId(j) 

        self.numPolys = numPolys
        self.pointCoords = pointCoords
        self.connectivity = connectivity

        opt = pymeshopt.Pymeshopt(pointCoords.astype(np.float32), connectivity.astype(np.int32), np.empty((0,4),dtype=np.int32), 1., 1., 1., 1.)
        output = opt.pyremoveduplicatetris()
        if (np.shape(self.connectivity)[0] != np.shape(output)[0]):
            print('Duplicate tris have been removed')

        self.connectivity = self.connectivity[np.unique(output)]
        self.pointCoords[:,[0,1,2]] = self.pointCoords[:,order]

        # uni = np.unique(output)
        # ori = np.arange(output.shape[0])
        # dup = np.setdiff1d(ori, uni)
        # duptri = connectivity[dup]
        # self.dupvert = np.unique(duptri).astype(np.int32)
        # print(self.dupvert) 
        
        if plot:
            print('Plotting OML triangular mesh')
            mesh = vedo.Mesh([self.pointCoords,self.connectivity])#, alpha=0.1
            mesh.backColor().lineColor('gree').lineWidth(3)
            plot = vedo.Plotter()
            plot.show('OML triangular mesh', mesh, viewup='z', axes=1, interactive = True)

    def buildup_octree(self):
        ''' Build up the octree structure.
        '''
        self.tree = ot.PyOctree(self.pointCoords,self.connectivity,self.quadlist)
        self.avg_size = self.tree.calculateavgedgesize()
        print('avg_size', self.avg_size)
        print()

    def add_member(self, member):
        ''' Add member to the aircraft model and update the list of members. mem_name dictionary
        ----------
        member : instance 
            An instance of Member class.
        '''
        self.members.append(member)
        self.num_members += 1
        #self.mem_name.update({member['name']:self.num_mem})        

    def compute_projection(self, plot = False):
        ''' Compute projection points and reconnect using  (node movement) .      
        ----------
        plot : bool
            If True, allow vedo plotting of OML triangular mesh.
        '''
        if plot:
            tree_plot_ori = self.tree
            projected_plot_ori = np.empty((0,3))
            projected_plot = np.empty((0,3))
            projecting_plot = np.empty((0,3))
            mesh_plot_ori = vedo.Mesh([self.pointCoords, self.connectivity])
            mesh_plot_ori.backColor().lineColor('green').lineWidth(3)
        
        for i in range(self.num_members):
            if np.all(self.members[i].options['projection_direction']==0):# projection algorithm: find the closet point 
                #print(self.members[i].options['name'])
                print('TODO: projection algorithm: find the closet point')     
            else:
                projection_points = self.members[i].options['points_to_be_projected']
                projection_direction = self.members[i].options['projection_direction']
                print('Start to project: ', i, self.members[i].options['name'])
                # print('projection_points',np.shape(projection_points))
                # print('projection_direction',np.shape(projection_direction))
                
                #num_points = np.shape(projection_points)[1]
                if  np.shape(projection_points)[1] == 2:
                    num_points = int(np.linalg.norm(projection_points[0,1,:] - projection_points[0,0,:])/self.avg_size)*5
                    print('num_points',num_points)
                    points_upper = np.linspace(projection_points[0,0,:], projection_points[0,1,:], num = num_points)
                    points_lower = np.linspace(projection_points[1,0,:], projection_points[1,1,:], num = num_points)
                    projection_points = np.stack((points_upper, points_lower)) 
                    print('projection_points',np.shape(projection_points))
                if len(np.shape(projection_direction)) == 2:
                    projection_direction0 = np.array((projection_direction[0,:]),ndmin=2)
                    projection_direction0 = np.repeat(projection_direction0,num_points,axis = 0)
                    projection_direction1 = np.array((projection_direction[1,:]),ndmin=2)
                    projection_direction1 = np.repeat(projection_direction1,num_points,axis = 0)
                    projection_direction = np.stack((projection_direction0, projection_direction1)) 
                    print('projection_direction',np.shape(projection_direction))
                print('--------')
                for j in range(2):
                    # if j == 1:
                    #     pass
                    projection_pts = projection_points[j,:,:]
                    projection_dirts = projection_points[j,:,:] + projection_direction[j,:,:]   

                    # _, proj_p= tree_plot_ori.ProjectPoints(projection_pts.astype(np.float32),projection_dirts.astype(np.float32))
                    # projected_plot = np.concatenate((projected_plot,proj_p))
                    # print('proj_p',len(proj_p)) 
                    # print(np.array(proj_p)) 

                    if True:              
                        reconnection= self.tree.ProjectandReconnect(projection_pts.astype(np.float32),projection_dirts.astype(np.float32)) 
                        self.pointCoords = reconnection.vertlist
                        self.connectivity = reconnection.trilist.astype(np.int32)
                        self.projected.append(reconnection.projlist)
                        self.members[i].options['projected_point_ids'].append(reconnection.projlist)
                        self.tree = ot.PyOctree(self.pointCoords,self.connectivity,self.quadlist)
                    
                        if plot:
                            # if j == 0:
                            #     pass
                            # else:
                            _, proj_p= tree_plot_ori.ProjectPoints(projection_pts.astype(np.float32),projection_dirts.astype(np.float32))
                            projected_plot = np.concatenate((projected_plot,proj_p))
                            print('proj_p',len(proj_p)) 
                            #print(np.array(proj_p)) 
                            projecting_plot = np.concatenate((projecting_plot, projection_pts))  
                            for k in reconnection.projlist:
                                projected_plot_ori = np.concatenate((projected_plot_ori,self.pointCoords[k,:].reshape((1,3))))
                #exit()
        print('')
        if plot: 
            projected_plot_update = np.empty((0,3), dtype=np.int32) 
            projected_total = np.array([], dtype=np.int32)
            for i in range(len(self.projected)):
                projected_total = np.concatenate((projected_total,self.projected[i])) 
            for k in range(len(projected_total)): 
                projected_plot_update = np.concatenate((projected_plot_update,self.pointCoords[projected_total[k],:].reshape((1,3))))
            mesh = vedo.Mesh([self.pointCoords, self.connectivity])#, alpha=0.1
            mesh.backColor().lineColor('green').lineWidth(3)
            points1 = vedo.Points(projected_plot_update, r= 15, c='red')
            plot = vedo.Plotter()
            plot.show('Updated OML + projection', mesh, points1, axes=1, interactive = False)            
            plot2 = vedo.Plotter()
            points2 = vedo.Points(projected_plot, r= 8, c='blue')
            points3 = vedo.Points(projecting_plot, r= 8, c='pink')
            points4 = vedo.Points(projected_plot_ori, r= 10, c='red')
            #Lines1 = vedo.Lines(projecting_plot,projection_dirts, c= 'black', lw=3)#,Lines1, points3
            plot2.show('Original OML + projection', mesh_plot_ori, points2, points4, axes=1, interactive = False)  

    def create_projection_members(self, plot = False):
        for i in range(self.num_members):
            ids = self.members[i].options['projected_point_ids'][0]
            _, indices = np.unique(ids,return_index=True)
            self.members[i].options['projected_point_ids'][0] = ids[np.sort(indices)]
            l0_id = self.members[i].options['projected_point_ids'][0]
            l0 = self.pointCoords[l0_id,:]
            print('l0',l0.shape)
            
            ids = self.members[i].options['projected_point_ids'][1]
            _, indices = np.unique(ids,return_index=True)
            self.members[i].options['projected_point_ids'][1] = ids[np.sort(indices)]
            # revrese l2
            l2_id = self.members[i].options['projected_point_ids'][1]
            l2 = self.pointCoords[l2_id,:]
            print('l2',l2.shape)

            num_v = 4
            LHS_uv = self.LHS(l0.shape[0],l2.shape[0],num_v, plot = True)#self.LHS(15,5,num_v, plot = True)#
            print('LHS_uv',LHS_uv.shape)

            l1 = np.linspace(l0[0,:], l2[0,:],num_v)
            l3 = np.linspace(l0[-1,:], l2[-1,:],num_v)
            print('l1',l1.shape)
            print('l3',l3.shape)
            vertcoord = self.TFI(l0, l1, l2, l3, LHS_uv, plot = True)

    def LHS(self, num_u0, num_u1, num_v, plot = False):

        if num_u0 > num_u1:
            num_u_max, num_u_min = num_u0,  num_u1
        else: 
            num_u_max, num_u_min = num_u1,  num_u0
        print('num_u_max',num_u_max,'num_u_min',num_u_min)

        num_vv = num_v
        vv = np.linspace(0, 1, num=num_vv)
        num_sampling = np.round(np.linspace(num_u_min, num_u_max+0.49, num_vv-1)).astype(int)
        print('num_sampling',num_sampling)
        ve_internal_pts = []
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
            uv = np.concatenate((uv,x))
            ve_internal_pts.append(vedo.Points(x, r= 13, c=i))

        if plot:
            u0 = np.vstack((np.linspace(0, 1, num=num_u0),np.linspace(0, 0, num=num_u0))).T
            u1 = np.vstack((np.linspace(0, 1, num=num_u1),np.linspace(1, 1, num=num_u1))).T
            v0 = np.vstack((np.linspace(0, 0, num=num_v),np.linspace(0, 1, num=num_v))).T
            v1 = np.vstack((np.linspace(1, 1, num=num_v),np.linspace(0, 1, num=num_v))).T
            ve_u0 = vedo.Points(u0, r= 15, alpha = 0.5, c='black')
            ve_u1 = vedo.Points(u1, r= 15, alpha = 0.5, c='black')
            ve_v0 = vedo.Points(v0, r= 15, alpha = 0.5, c='black')
            ve_v1 = vedo.Points(v1, r= 15, alpha = 0.5, c='black')
            plot = vedo.Plotter()
            plot.show('Boundary', ve_internal_pts, ve_u0,  ve_v0, ve_u1, ve_v1, axes=1, interactive=False) 
        
        return uv
            
    def TFI(self, l0, l1, l2, l3, uv, plot = False):
        tck_l0, u_l0 = interpolate.splprep(l0.T, k = 2, s=0)
        new_points = interpolate.splev(np.array([0.5,0.7]), tck_l0)
        print('new_points',new_points)
        tck_l2, u_l2 = interpolate.splprep(l2.T, k = 2, s=0)
        tck_l1, u_l1 = interpolate.splprep(l1.T, k = 2, s=0)
        tck_l3, u_l3 = interpolate.splprep(l3.T, k = 2, s=0)
        u = uv[:,0]
        v = uv[:,1]
        point_01 = l0[0,:]
        point_23 = l2[-1,:]
        point_03 = l0[-1,:]
        point_21 = l2[0,:]
        print(u.shape, point_01.shape)
        # print(np.outer(u,point_01).shape)
        print(((1-v)*np.array(interpolate.splev(u,tck_l0))[0,:]).shape)
        print(((1-u)*(1-v)*point_01[0]).shape)
        vertcoord = np.zeros((uv.shape[0],3))
        vertcoord[:,0] = (1-v)*np.array(interpolate.splev(u,tck_l0))[0,:] + v*np.array(interpolate.splev(u,tck_l2))[0,:] +(1-u)*np.array(interpolate.splev(v,tck_l1))[0,:] + u*np.array(interpolate.splev(v,tck_l3))[0,:]\
            -((1-u)*(1-v)*point_01[0] + u*v*point_23[0] + u*(1-v)*point_03[0] + (1-u)*v*point_21[0])
        vertcoord[:,1] = (1-v)*np.array(interpolate.splev(u,tck_l0))[1,:] + v*np.array(interpolate.splev(u,tck_l2))[1,:] +(1-u)*np.array(interpolate.splev(v,tck_l1))[1,:] + u*np.array(interpolate.splev(v,tck_l3))[1,:]\
            -((1-u)*(1-v)*point_01[1] + u*v*point_23[1] + u*(1-v)*point_03[1] + (1-u)*v*point_21[1])
        vertcoord[:,2] = (1-v)*np.array(interpolate.splev(u,tck_l0))[2,:] + v*np.array(interpolate.splev(u,tck_l2))[2,:] +(1-u)*np.array(interpolate.splev(v,tck_l1))[2,:] + u*np.array(interpolate.splev(v,tck_l3))[2,:]\
            -((1-u)*(1-v)*point_01[2] + u*v*point_23[2] + u*(1-v)*point_03[2] + (1-u)*v*point_21[2])
        print('vertcoord',vertcoord.shape)
        if plot:   
            ve_l0 = vedo.Points(l0, r = 15, alpha = 0.5, c='black')
            ve_l1 = vedo.Points(l1, r = 15, alpha = 0.5, c='green')
            ve_l2 = vedo.Points(l2, r = 15, alpha = 0.5, c='yellow')
            ve_l3 = vedo.Points(l1, r = 15, alpha = 0.5, c='pink')
            ve_internal_pts = vedo.Points(vertcoord, r = 10, c='red')
            plot = vedo.Plotter()
            plot.show('Boundary', ve_internal_pts, ve_l0,  ve_l1, ve_l2, ve_l3, axes=1, interactive=True) 
        return vertcoord