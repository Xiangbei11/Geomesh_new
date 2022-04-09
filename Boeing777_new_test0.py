import numpy as np
from geomesh_new import Geomesh
from member_new import Member
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time



import vedo

# ---- Set path ----
aircraft = Geomesh(path_of_file = 'CAD_new/Boeing_777-9x_9236_notwatertight.stl')

# ---- Define topology and octree ----
aircraft.compute_topology(plot = False) 
aircraft.buildup_octree()

# ---- Define spars ----
x_leading_root = 29.6
x_leading_tip = 47.6
y_leading_root = 4.
y_leading_tip = 33.
z1 = 30.
z2 = -30.
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

# print(points_to_be_projected)
# print(points_to_be_projected[0,1,:])
# print(np.shape(points_to_be_projected))
# print(len(points_to_be_projected))
# print(len(np.shape(points_to_be_projected)))
# exit()
# ---- Define ribs ----
num_ribs = 0
x_start_list = [4472/145,]# 32.4, 33.6, 34.8, 36, 37.2, 38.4, 39.6, 40.8, 42, 43.2, 44.4, 45.6, 46.8]
x_end_list = [x/61 for x in [2191,]]# 2247, 2303, 2359, 2415, 2471, 2527, 2583, 2639, 2695, 2751, 2807, 2863, 2919]] 
y_list = [6,]# 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
z1 = 5
z2 = -5
projection_direction = np.array(([0, 0, -1],[0, 0, 1]))
num_ribs = 0
num_points = 50
for x_start, x_end, y in zip(x_start_list, x_end_list, y_list):
    point_start_upper = np.array([x_start,y,z1])
    point_end_upper = np.array([x_end,y,z1])
    point_start_lower = np.array([x_start,y,z2])
    point_end_lower = np.array([x_end,y,z2])
    points_upper = np.stack((point_start_upper, point_end_upper))
    points_lower = np.stack((point_start_lower, point_end_lower))
    points_to_be_projected = np.stack((points_upper, points_lower))
    aircraft.add_member(Member(
        name = 'rib_{}'.format(num_ribs),
        points_to_be_projected = points_to_be_projected,
        projection_direction = projection_direction,
        ))
    num_ribs += 1
    if num_ribs ==1:
        break

# ---- Projection ----
start = time.time()
aircraft.compute_projection(plot = True)
end = time.time()
print('Time: ',end - start)
aircraft.create_projection_members()

exit()
# ---- Just test the path of gurobi licence file ----
import gurobipy as gp
test = gp.Model('test') # export GRB_LICENSE_FILE=/Users/Sansara/Public/Code/A/gurobi.lic