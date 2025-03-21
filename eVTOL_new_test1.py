import vedo

from geomesh_new import Geomesh
from member_new import Member

import sys
sys.path.insert(0,'/Users/Sansara/Public/Code/Geomesh/Geomesh_new/lsdo_kit')
from lsdo_kit.design.design import Design

''' Creating instances of lsdo_kit design class '''
design = Design('CAD_new/eVTOL.stp')

geo = design.design_geometry 



#TODO Change Geomesh to ShellMesh/UnstruturedQuadMesh but maybe keep it as a seperate github package
#TODO ShellMesh will inherit the Mesh class from lsdo_kit.simulation.mesh.mesh and follow the same design principles
from shellmesh import ShellMesh

design_geometry_class = geo 
shell_mesh = ShellMesh('shell_mesh', geo)
#TODO shell_mesh.add_mesh(rib_pointset) / shell_mesh.assemble()
shell_solver = ShellSolver('shell_solver', mesh=shell_mesh)

#TODO Update eVTOL test case also with lsdo_kit (Test driven development)

#TODO Keep and publish pymeshopt as aseperate package

# ---- Set path ----
aircraft = Geomesh(path_of_file = 'CAD_new/eVTOL.stl')#25362

# ---- Define topology and octree ----
aircraft.compute_topology(plot = False) 
aircraft.buildup_octree()


x_leading_root = 16
x_leading_tip = -15
y_leading_root = 20
y_leading_tip = 209
x_trailing_root = 46
x_trailing_tip = 6
y_trailing_root = y_leading_root
y_trailing_tip = y_leading_tip

z1 = 10
z2 = -10
num_rib = 14

# # ---- Define spars ----
# num_spar_sections = 0
# x_leading_root_list = np.linspace(x_leading_root,x_leading_tip,num_rib).take([0,2,3,4,5,6,7,8,9,10,11,12])
# x_leading_tip_list = np.linspace(x_leading_root,x_leading_tip,num_rib).take([2,3,4,5,6,7,8,9,10,11,12,13])
# y_leading_root_list = np.linspace(y_leading_root,y_leading_tip,num_rib).take([0,2,3,4,5,6,7,8,9,10,11,12])
# y_leading_tip_list = np.linspace(y_leading_root,y_leading_tip,num_rib).take([2,3,4,5,6,7,8,9,10,11,12,13])
# for x_leading_root, x_leading_tip, y_leading_root, y_leading_tip in zip(x_leading_root_list, x_leading_tip_list, y_leading_root_list, y_leading_tip_list):
#     point_start_upper = np.array([x_leading_root,y_leading_root,z1])
#     point_end_upper = np.array([x_leading_tip,y_leading_tip,z1])
#     point_start_lower = np.array([x_leading_root,y_leading_root,z2])
#     point_end_lower = np.array([x_leading_tip,y_leading_tip,z2])
#     points_upper = np.stack((point_start_upper, point_end_upper))#points_upper = np.linspace(point_start_upper, point_end_upper, num = 2)
#     points_lower = np.stack((point_start_lower, point_end_lower))#points_lower = np.linspace(point_start_lower, point_end_lower, num = 2)
#     points_to_be_projected = np.stack((points_upper, points_lower))
#     projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
#     aircraft.add_member(Member(
#         name = 'spar_{}'.format(num_spar_sections),
#         points_to_be_projected = points_to_be_projected,
#         projection_direction = projection_direction,
#         ))
#     num_spar_sections += 1


# ---- Define ribs ----
num_ribs = 0
x_leading_root = 16
x_leading_tip = -15
y_leading_root = 20
y_leading_tip = 209
x_trailing_root = 46
x_trailing_tip = 6
y_trailing_root = y_leading_root
y_trailing_tip = y_leading_tip
x_l_list = np.linspace(x_leading_root,x_leading_tip,num_rib)[1:-1]
x_t_list = np.linspace(x_trailing_root,x_trailing_tip,num_rib)[1:-1]
y_list = np.linspace(y_trailing_root,y_trailing_tip,num_rib)[1:-1]
for x_l, x_t, y in zip(x_l_list, x_t_list, y_list):
    point_start_upper = np.array([x_l,y,z1])
    point_end_upper = np.array([x_t,y,z1])
    point_start_lower = np.array([x_l,y,z2])
    point_end_lower = np.array([x_t,y,z2])
    points_upper = np.stack((point_start_upper, point_end_upper))
    points_lower = np.stack((point_start_lower, point_end_lower))
    points_to_be_projected = np.stack((points_upper, points_lower))
    projection_direction = np.array(([0., 0., -2.],[0., 0., 2.]))
    num_ribs += 1
    if num_ribs == 2:
        aircraft.add_member(Member(
        name = 'rib_{}'.format(num_ribs),
        points_to_be_projected = points_to_be_projected,
        projection_direction = projection_direction,
        ))


# point_start_upper = np.array([x_trailing_root,y_trailing_root,z1])
# point_end_upper = np.array([x_trailing_tip,y_trailing_tip,z1])
# point_start_lower = np.array([x_trailing_root,y_trailing_root,z2])
# point_end_lower = np.array([x_trailing_tip,y_trailing_tip,z2])
# points_upper = np.stack((point_start_upper, point_end_upper))
# points_lower = np.stack((point_start_lower, point_end_lower))
# points_to_be_projected = np.stack((points_upper, points_lower))
# projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
# spar_2 = Member(
#     name = 'spar_{}'.format(num_spar_sections),
#     points_to_be_projected = points_to_be_projected,
#     projection_direction = projection_direction,
#     )
# num_spar_sections += 1
# aircraft.add_member(spar_2)

aircraft.compute_projection(plot = True)
#print('num_spar_sections',num_spar_sections)
print('number of ribs', num_ribs)
aircraft.create_projection_members(plot = False)
# plot_empty = vedo.Plotter()
# plot_empty.show(interactive = True) 

exit()



 







x_trailing_root_tt = (x_leading_root+x_trailing_root)/2
x_trailing_tip_tt = (x_leading_tip+x_trailing_tip)/2
y_trailing_root_tt = y_leading_root
y_trailing_tip_tt = y_leading_tip
# point_start_upper = np.array([x_trailing_root_tt,y_trailing_root_tt,z1])
# point_end_upper = np.array([x_trailing_tip_tt,y_trailing_tip_tt,z1])
# point_start_lower = np.array([x_trailing_root_tt,y_trailing_root_tt,z2])
# point_end_lower = np.array([x_trailing_tip_tt,y_trailing_tip_tt,z2])
# points_upper = np.stack((point_start_upper, point_end_upper))
# points_lower = np.stack((point_start_lower, point_end_lower))
# points_to_be_projected = np.stack((points_upper, points_lower))
# projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
# num_spar_sections = 0
# spar_tt = Member(
#     name = 'spar_tt',
#     points_to_be_projected = points_to_be_projected,
#     projection_direction = projection_direction,
#     )
# aircraft.add_member(spar_tt)



