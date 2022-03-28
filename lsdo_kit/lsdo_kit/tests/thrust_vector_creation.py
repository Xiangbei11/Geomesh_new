import csdl
import numpy as np
from lsdo_kit.design.design_geometry.utils.generate_ffd import create_ffd
from lsdo_kit.design.design_geometry.core.ffd import FFD
from lsdo_kit.design.design_geometry.design_geometry import DesignGeometry 
from lsdo_kit.old_files.mesh import Mesh
from lsdo_kit.design.design_geometry.core.component import Component

def generate_thrust_vector(geo, nacelle_comp):


    up_direction = np.array([0., 0., 1.])
    down_direction = np.array([0., 0., -1.])

    _nacelle_length = nacelle_comp.x_max - nacelle_comp.x_min
    _quarter_nacelle_length = (1/4) * _nacelle_length
    _tip_indices = np.argwhere(nacelle_comp.embedded_entities_control_points == nacelle_comp.x_min)
    _tip_index = _tip_indices[0,0]

    _nacelle_z_max = nacelle_comp.z_max
    _nacelle_z_min = nacelle_comp.z_min

    thrust_tip_control_point = nacelle_comp.embedded_entities_control_points[_tip_index, :]

    thrust_top_origin_control_point = thrust_tip_control_point
    thrust_top_origin_control_point[0] = thrust_top_origin_control_point[0] + _quarter_nacelle_length
    thrust_top_origin_control_point[2] = _nacelle_z_max

    thrust_bot_origin_control_point = thrust_tip_control_point
    thrust_bot_origin_control_point[0] = thrust_top_origin_control_point[0] + _quarter_nacelle_length
    thrust_bot_origin_control_point[2] = _nacelle_z_min

    thrust_top_origin, thrust_top_origin_coord = geo.project_points(thrust_top_origin_control_point, projection_direction = down_direction, projection_targets_names=[nacelle_comp.name])
    thrust_bot_origin, thrust_bot_origin_coord = geo.project_points(thrust_bot_origin_control_point, projection_direction = up_direction, projection_targets_names=[nacelle_comp.name])
    thrust_origin = geo.perform_linear_interpolation(thrust_top_origin, thrust_bot_origin,[1], output_parameters = np.array([0.5]))

    thrust_tip, thrust_tip_coord  = geo.project_points(thrust_tip_control_point, projection_direction = down_direction, projection_targets_names=[nacelle_comp.name])
    
    thrust_vector = geo.subtract_pointsets(thrust_tip, thrust_origin)

    geo.assemble()
    geo.evaluate()

    thrust_mag = np.linalg.norm(thrust_vector.physical_coordinates)
    thrust_vector_norm = geo.divide_pointset_by_scalar(thrust_vector, thrust_mag)

    return [thrust_origin, thrust_vector_norm]
