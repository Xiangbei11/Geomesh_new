import numpy as np
from geomesh_new import Geomesh
from member_new import Member
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import vedo

# ---- Set path ----
aircraft = Geomesh('CAD_new/eVTOL_3260.stl')#25362

# ---- Define topology and octree ----
aircraft.compute_topology(plot = False) 
aircraft.buildup_octree()

# ---- Define spars ----
x_leading_root = 16
x_leading_tip = -15
y_leading_root = 20
y_leading_tip = 209
z1 = 30
z2 = -5
point_start_upper = np.array([x_leading_root,y_leading_root,z1])
point_end_upper = np.array([x_leading_tip,y_leading_tip,z1])
point_start_lower = np.array([x_leading_root,y_leading_root,z2])
point_end_lower = np.array([x_leading_tip,y_leading_tip,z2])
points_upper = np.stack((point_start_upper, point_end_upper))#points_upper = np.linspace(point_start_upper, point_end_upper, num = 2)
points_lower = np.stack((point_start_lower, point_end_lower))#points_lower = np.linspace(point_start_lower, point_end_lower, num = 2)
points_to_be_projected = np.stack((points_upper, points_lower))
projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
num_spar_sections = 0
spar_1 = Member(
    name = 'spar_{}'.format(num_spar_sections),
    points_to_be_projected = points_to_be_projected,
    projection_direction = projection_direction,
    )
num_spar_sections += 1
aircraft.add_member(spar_1)
aircraft.compute_projection(plot = True)
exit()
x_trailing_root = 46
x_trailing_tip = 6
y_trailing_root = y_leading_root
y_trailing_tip = y_leading_tip
# point_start_upper = np.array([x_trailing_root,y_trailing_root,z1])
# point_end_upper = np.array([x_trailing_tip,y_trailing_tip,z1])
# point_start_lower = np.array([x_trailing_root,y_trailing_root,z2])
# point_end_lower = np.array([x_trailing_tip,y_trailing_tip,z2])
# points_upper = np.stack((point_start_upper, point_end_upper))
# points_lower = np.stack((point_start_lower, point_end_lower))
# points_to_be_projected = np.stack((points_upper, points_lower))
# projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
# num_spar_sections = 0
# spar_2 = Member(
#     name = 'spar_{}'.format(num_spar_sections),
#     points_to_be_projected = points_to_be_projected,
#     projection_direction = projection_direction,
#     )
# num_spar_sections += 1
# aircraft.add_member(spar_2)

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

# ---- Define ribs ----
num_ribs = 0
num_rib = 6
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
    projection_direction = np.array(([0., 0., -1.],[0., 0., 1.]))
    num_spar_sections = 0
    aircraft.add_member(Member(
        name = 'rib_{}'.format(num_ribs),
        points_to_be_projected = points_to_be_projected,
        projection_direction = projection_direction,
        ))
    num_ribs += 1
print('number of ribs', num_ribs)

