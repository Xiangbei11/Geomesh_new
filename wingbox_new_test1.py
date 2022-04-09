import vedo

from geomesh_new import Geomesh
from member_new import Member

import sys
sys.path.insert(0,'/Users/Sansara/Public/Code/Geomesh/Geomesh_new/lsdo_kit')
from lsdo_kit.design.design import Design

''' Creating instances of lsdo_kit design class '''
design = Design('CAD_new/uCRM-9_wingbox.stp')

geo = design.design_geometry 
# print(type(geo)) #<class 'lsdo_kit.design.design_geometry.design_geometry.DesignGeometry'>
# print(type(geo.output_bspline_entity_dict.values())) lsdo_kit.design.design_geometry.bsplines.bspline_surface.BSplineSurface object

'''Happened in design_geometry class'''
#TODO Indtify the upper and lower surfaces of wingbox as OML/aircraft geometry
#   #TODO (i!=3 and i!=2), (i!=6 and i!=7): Surface (2 and 3), (6 and 7) is completely(?) overlapping
#   #TODO There are also partly overlapping between upper/lower surfaces
#TODO For each of the ribs
#   #TODO There are also intersections between ribs.
#   1) Extract the upper and lower curves. 
#   2) Discretize those curves
#   3) Do projection of these curves on OML

#TODO Change Geomesh to ShellMesh/UnstruturedQuadMesh but maybe keep it as a seperate github package
#TODO ShellMesh will inherit the Mesh class from lsdo_kit.simulation.mesh.mesh and follow the same design principles
from shellmesh import ShellMesh

design_geometry_class = geo 
shell_mesh = ShellMesh('shell_mesh', geo)
#TODO shell_mesh.add_mesh(rib_pointset) / shell_mesh.assemble()
shell_solver = ShellSolver('shell_solver', mesh=shell_mesh)

#TODO Update eVTOL test case also with lsdo_kit (Test driven development)

#TODO Keep and publish pymeshopt as aseperate package

plot_empty = vedo.Plotter()
plot_empty.show(interactive = True) 