import numpy as np
from lsdo_kit.problem import Problem
from lsdo_kit.design.design import Design
from lsdo_kit.simulation.cruise_simulation import Simulation, CruiseSimulation
from lsdo_kit.design.design_geometry.core.component import Component
from lsdo_kit.design.design_geometry.tests.organized_test.mesh_creation_script import generate_meshes
from lsdo_kit.design.design_geometry.core.geometric_outputs import GeometricOuputs
from lsdo_kit.design.design_geometry.core.actuation import Actuation
from lsdo_kit.simulation.mesh.vlm_mesh import VLMMesh
from lsdo_kit.simulation.solver.vlm_solver import VlmSolver
from lsdo_kit.design.design_geometry.tests.organized_test.mesh_creation_script import generate_meshes
from lsdo_kit.tests.thrust_vector_creation import generate_thrust_vector
from vedo import Points, Plotter, LegendBox
import matplotlib.pyplot as plt
import timeit


''' Creating File path for Rect Wing DesignGeometry'''
path_name = '../design/design_geometry/examples/CAD/'
file_name = 'both_wing_all_nacelles.stp'

''' Creating instances of problem and design class '''
problem = Problem()

design = Design(path_name + file_name)
geo = design.design_geometry

nt = 5

velocity = np.tile(np.array([50, 0, 1]), (5,1))
test_sim = CruiseSimulation('test_sim', velocity=velocity, density='pls_work') 
test_sim.nt = 5

''' Creating a component and adding to design '''
# fuselage = Component(stp_entity_names=['Fuselage'], name='fuselage')
# # fuselage.add_shape_parameter(property_name = 'rot_x', parameter_name='linear', order=1, num_cp=2, dv=False, val=np.zeros(2))
# design.add_component(fuselage)

fwing = Component(stp_entity_names=['FrontWing'], name='front_wing')
design.add_component(fwing)

# flrotor1 = Component(stp_entity_names=['FrontLeftRotor1'], name='flrotor1')
# design.add_component(flrotor1)

# flrotor2 = Component(stp_entity_names=['FrontLeftRotor2'], name='flrotor2')
# design.add_component(flrotor2)

# flrotor3 = Component(stp_entity_names=['FrontLeftRotor3'], name='flrotor3')
# design.add_component(flrotor3)

# frrotor1 = Component(stp_entity_names=['FrontRightRotor1'], name='frrotor1')
# design.add_component(frrotor1)

# frrotor2 = Component(stp_entity_names=['FrontRightRotor2'], name='frrotor2')
# design.add_component(frrotor2)

# frrotor3 = Component(stp_entity_names=['FrontRightRotor3'], name='frrotor3')
# design.add_component(frrotor3)

flnacelle1 = Component(stp_entity_names=['FrontLeftNacelle1'], name='flnacelle1')
design.add_component(flnacelle1)
flnacelle1_thrust = generate_thrust_vector(geo, flnacelle1)

# print('MAX NACELLE POS: ', flnacelle1.x_min)
# print('NACELLE EMBEDDED POINTS: ', flnacelle1.embedded_entities_control_points)
# print('MAX ROTOR POS: ', flrotor1.x_max)
# print('MIN ROTOR POS: ', flrotor1.x_min)

# for name in flnacelle1.embedded_entity_names:
#     surface = geo.input_bspline_entity_dict[name]
#     vp_init = Plotter()
#     vps1 = Points(surface.control_points, r=8, c = 'blue')
#     # vps.append(vps2)
#     vp_init.show(vps1, f'{surface.name}', axes=1, viewup="z", interactive = True)

# for name in flrotor1.embedded_entity_names:
#     surface = geo.input_bspline_entity_dict[name]
#     vp_init = Plotter()
#     vps1 = Points(surface.control_points, r=8, c = 'blue')
#     # vps.append(vps2)
#     vp_init.show(vps1, f'{surface.name}', axes=1, viewup="z", interactive = True)

flnacelle2 = Component(stp_entity_names=['FrontLeftNacelle2'], name='flnacelle2')
design.add_component(flnacelle2)
flnacelle2_thrust = generate_thrust_vector(geo, flnacelle2)

flnacelle3 = Component(stp_entity_names=['FrontLeftNacelle3'], name='flnacelle3')
design.add_component(flnacelle3)
flnacelle1_thrust = generate_thrust_vector(geo, flnacelle3)

frnacelle1 = Component(stp_entity_names=['FrontRightNacelle1'], name='frnacelle1')
design.add_component(frnacelle1)
flnacelle1_thrust = generate_thrust_vector(geo, frnacelle1)

frnacelle2 = Component(stp_entity_names=['FrontRightNacelle2'], name='frnacelle2')
design.add_component(frnacelle2)
flnacelle1_thrust = generate_thrust_vector(geo, frnacelle1)

frnacelle3 = Component(stp_entity_names=['FrontRightNacelle3'], name='frnacelle3')
design.add_component(frnacelle3)
flnacelle1_thrust = generate_thrust_vector(geo, frnacelle1)

# rwing = Component(stp_entity_names=['RearWing'], name='Rear_wing')
# design.add_component(rwing)

# rlrotor = Component(stp_entity_names=['LeftRearRotor'], name='left_rear_rotor')
# design.add_component(rlrotor)

# rrrotor = Component(stp_entity_names=['RightRearRotor'], name='right_rear_rotor')
# design.add_component(rrrotor)

# rlnacelle = Component(stp_entity_names=['LeftRearNacelle'], name='left_rear_nacelle')
# design.add_component(rlnacelle)

# rrnacelle = Component(stp_entity_names=['RightRearNacelle'], name='right_rear_nacelle')
# design.add_component(rrnacelle)

# tail = Component(stp_entity_names=['Tail'], name='tail')
# design.add_component(tail)

# front_landing_fairing = Component(stp_entity_names=['FrontLandingFairing'], name='front_landing_fairing')
# design.add_component(front_landing_fairing)

# main_landing_fairing = Component(stp_entity_names=['MainLandingFairing'], name='main_landing_fairing')
# design.add_component(main_landing_fairing)

# front_tires = Component(stp_entity_names=['FrontTires'], name='front_tires')
# design.add_component(front_tires)

# main_tires = Component(stp_entity_names=['MainTires'], name='main_tires')
# design.add_component(main_tires)


''' Defining Points and directions for projections '''
up_direction = np.array([0., 0., 1.])
down_direction = np.array([0., 0., -1.])
mm2ft = 304.8

''' Front wing corner points '''
front_left_lead_point = np.array([3962.33, 6662.76, 2591])/mm2ft
front_left_trail_point = np.array([4673.88, 6662.76, 2590.8])/mm2ft
front_right_lead_point = np.array([3962.33, -6662.76, 2591])/mm2ft
front_right_trail_point = np.array([4673.88, -6662.76, 2590.8])/mm2ft

front_fuselage = np.array([3600.276, 453.0, 2600])/mm2ft
back_fuselage = np.array([3600.276, -453.0, 2600])/mm2ft


''' Project points '''
front_left_lead_point, front_left_lead_point_coord = design.project_points(front_left_lead_point, projection_direction = down_direction, projection_targets_names=["front_wing"])
front_left_trail_point ,front_left_trail_point_coord = design.project_points(front_left_trail_point, projection_direction = down_direction, projection_targets_names=["front_wing"])

front_right_lead_point ,front_right_lead_point_coord = design.project_points(front_right_lead_point, projection_direction = down_direction, projection_targets_names=["front_wing"])
front_right_trail_point ,front_right_trail_point_coord = design.project_points(front_right_trail_point, projection_direction = down_direction, projection_targets_names=["front_wing"])

front_fuse, front_fuse_coord = design.project_points(front_fuselage, projection_direction = down_direction, projection_targets_names=["front_wing"])
back_fuse, back_fuse_coord = design.project_points(back_fuselage, projection_direction = down_direction, projection_targets_names=["front_wing"])

''' Creating a pointing vector across the wing for rotation '''
output_parameters = np.array([0.75])
quarter_left = design.perform_linear_interpolation(pointset_start=front_left_lead_point, pointset_end=front_left_trail_point, shape=(1,), output_parameters=output_parameters)
quarter_right = design.perform_linear_interpolation(pointset_start=front_right_lead_point, pointset_end=front_right_trail_point, shape=(1,), output_parameters=output_parameters)

''' Define the actuation profile for the FRONT wings and rotors '''
actuation_profile = np.linspace(0, -np.pi/2, 5)
# actuation_profile = np.zeros((3,))
name = 'multiple_timesteps'
actuating_comps = [fwing, flnacelle1, flnacelle2, flnacelle3, frnacelle1, frnacelle2, frnacelle3]
actuation_obj = Actuation(name='multi_wing_rot', actuation_profile=actuation_profile, origin=front_fuse, pointset2=back_fuse, actuating_components=actuating_comps)
test_sim.add_actuations(design=design, actuation_list=[actuation_obj])
# test_sim.add_actuations(design=design, actuation_list=[actuation_obj1])


''' Add Design to the Problem class and call the ProblemModel '''


geo.assemble()
geo.evaluate()

problem.set_design(design)
problem.add_simulation(test_sim)

problem.assemble()
problem.run()


# vp = Plotter()
# for t in range(180):
#     vps = Points(problem.sim['simulation_models.test_sim.actuation_model.actuated_control_points'][t,:,:], r=8, c = 'red')
#     vp.show(vps, f'Actuation_{t}', axes=1, viewup="z", interactive=False)
#     vp.screenshot(f'actuation{t}')



# ''' Rear wing corner points'''
# rear_left_lead_point = np.array([8277.012, 1536.289, 4005.94])/mm2ft
# rear_left_trail_point = np.array([9123.56, 1536.289, 4005.94])/mm2ft
# rear_right_lead_point = np.array([8277.012, -1536.289, 4005.94])/mm2ft
# rear_right_trail_point = np.array([9123.56, -1536.289, 4005.94])/mm2ft



# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# x = geo.total_cntrl_pts_vector[:,0]
# y = geo.total_cntrl_pts_vector[:,1]
# z = geo.total_cntrl_pts_vector[:,2]
# ax.scatter(x, y, z, 'green')
# # ax.set_title(f'{surface.name}')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()


# print(geo.total_cntrl_pts_vector.shape)
# vp_init = Plotter()
# vps = []
# vps1 = Points(geo.total_cntrl_pts_vector, r=8, c = 'blue')
# # vps.append(vps2)
# vp_init.show(vps1, 'Total_control_points', axes=1, viewup="z", interactive = True)

# ''' Project points onto front wing '''
# front_wing_lead_left, front_wing_lead_left_coord = design.project_points(front_left_lead_point, projection_direction = down_direction, projection_targets_names=["front_wing"])
# front_wing_trail_left ,front_wing_trail_left_coord = design.project_points(front_left_trail_point, projection_direction = down_direction, projection_targets_names=["front_wing"])
# front_wing_lead_right ,front_wing_lead_right_coord = design.project_points(front_right_lead_point, projection_direction = down_direction, projection_targets_names=["front_wing"])
# front_wing_trail_right ,front_wing_trail_right_coord = design.project_points(front_right_trail_point, projection_direction = down_direction, projection_targets_names=["front_wing"])

# print(geo.total_cntrl_pts_vector.shape)

# for surface in geo.input_bspline_entity_dict.values():
# vp_init = Plotter()
# vps = []
# vps1 = Points(geo.total_cntrl_pts_vector[-1:3], r=8, c = 'blue')
# # vps.append(vps2)
# vp_init.show(vps1, 'Total_control_points', axes=1, viewup="z", interactive = True)




# for surface in geo.input_bspline_entity_dict.values():
#     fig = plt.figure()
#     ax = plt.axes(projection ='3d')
#     x = surface.control_points[:,0]
#     y = surface.control_points[:,1]
#     z = surface.control_points[:,2]
#     ax.scatter(x, y, z, 'green')
#     ax.set_title(f'{surface.name}')
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.show()


# for surface in geo.input_bspline_entity_dict.values():
#     vp_init = Plotter()
#     vps1 = Points(surface.control_points, r=8, c = 'blue')
#     # vps.append(vps2)
#     vp_init.show(vps1, f'{surface.name}', axes=1, viewup="z", interactive = True)


# print(geo.total_cntrl_pts_vector.shape)