import numpy as np
import vtk
import vedo

import pymeshopt
from pyoctree import pyoctree as ot
from member_new import Member

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# from options_dictionary import OptionsDictionary
# from meshopt import meshopt
# from scipy import interpolate
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
                    projection_pts = projection_points[j,:,:]
                    projection_dirts = projection_points[j,:,:] + projection_direction[j,:,:]                    
                    reconnection= self.tree.ProjectandReconnect(projection_pts.astype(np.float32),projection_dirts.astype(np.float32)) 
                    self.pointCoords = reconnection.vertlist
                    self.connectivity = reconnection.trilist.astype(np.int32)
                    self.projected.append(reconnection.projlist)
                    self.members[i].options['projected_point_ids'].append(reconnection.projlist)
                    self.tree = ot.PyOctree(self.pointCoords,self.connectivity,self.quadlist)
                    
                    if plot:
                        _, proj_p= tree_plot_ori.ProjectPoints(projection_pts.astype(np.float32),projection_dirts.astype(np.float32))
                        projected_plot = np.concatenate((projected_plot,proj_p))   
                        projecting_plot = np.concatenate((projecting_plot, projection_pts))  
                        for k in reconnection.projlist:
                            projected_plot_ori = np.concatenate((projected_plot_ori,self.pointCoords[k,:].reshape((1,3))))
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
            #points3 = vedo.Points(projecting_plot, r= 8, c='pink')
            points4 = vedo.Points(projected_plot_ori, r= 10, c='red')
            plot2.show('Original OML + projection', mesh_plot_ori, points2, points4, axes=1, interactive = True)  

    def create_projection_members(self):
        for i in range(self.num_members):
            ids = self.members[i].options['projected_point_ids'][0]
            _, indices = np.unique(ids,return_index=True)
            self.members[i].options['projected_point_ids'][0] = ids[np.sort(indices)]
            l0 = self.members[i].options['projected_point_ids'][0]
            
            ids = self.members[i].options['projected_point_ids'][1]
            _, indices = np.unique(ids,return_index=True)
            self.members[i].options['projected_point_ids'][1] = ids[np.sort(indices)]
            # revrese l2
            l2 = self.members[i].options['projected_point_ids'][1][::-1]



