#from lsdo_kit.design.design import Design
import sys
sys.path.insert(0,'/Users/Sansara/Public/Code/Geomesh/Geomesh_new/lsdo_kit')
from lsdo_kit.design.design import Design

''' Creating instances of lsdo_kit design class '''
#design = Design('CAD_new/uCRM-9_wingbox.stp')
#esign = Design('CAD_new/wing_gmsh.stp')
design = Design('CAD_new/wing_openvsp.stp')

geo = design.design_geometry
# for item in geo.initial_input_bspline_entity_dict:
#     print(item)

import vedo
plot_empty = vedo.Plotter()
plot_empty.show(interactive = True) 