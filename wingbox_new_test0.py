from geomesh_new import Geomesh
from member_new import Member

import vedo

import sys
sys.path.insert(0,'/Users/Sansara/Public/Code/Geomesh/Geomesh_new/lsdo_kit')
from lsdo_kit.design.design import Design

''' Creating instances of lsdo_kit design class '''
design = Design('CAD_new/uCRM-9_wingbox.stp')
#design = Design('CAD_new/wing_gmsh.stp')
#design = Design('CAD_new/wing_openvsp.stp')

geo = design.design_geometry 
# print(type(geo)) 
# # ##<class 'lsdo_kit.design.design_geometry.design_geometry.DesignGeometry'>
# print(type(geo.output_bspline_entity_dict.values())) #geo.output_geomesh_bspline_entity_dict
# # ##lsdo_kit.design.design_geometry.bsplines.bspline_surface.BSplineSurface object


# aircraft = Geomesh(lsdo_kit_design_geometry_class = geo)

# aircraft.compute_skin_triangulation(bspline_surface_index_list0) #aircraft
# #1) compute b-spline surface points 2) 2D constrained DT
# aircraft.buildup_octree()

# aircraft.add_projection_member(bspline_surface_index_list1)
# #aircraft.compute_projection()
# aircraft.create_internal_member()

# aircraft.meshskin(itr=[0,1,3],w1=0.5,w2=0.5,w3=0.5,plot=0,name='test_eVTOL/eVTOL_uCRM-9_test0')
# aircraft.meshmembers(itr=[0,1,3],w1=0.5,w2=0.5,w3=0.5,plot=0,name='test_eVTOL/eVTOL_uCRM-9_test0')
# aircraft.integratemeshes(vtk_file_name = 'test_uCRM-9/eVTOL_uCRM-9_test0') 

# plot_empty = vedo.Plotter()
# plot_empty.show(interactive = True) 