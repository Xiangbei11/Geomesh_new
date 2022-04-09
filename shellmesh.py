import sys
sys.path.insert(0,'/Users/Sansara/Public/Code/Geomesh/Geomesh_new/lsdo_kit')
from lsdo_kit.simulation.mesh.mesh import Mesh

class ShellMesh(Mesh):
    def __init__(self, name, design_geometry_class=[]) -> None:
        super().__init__(name, design_geometry_class)

if __name__ == '__main__':
    from shellmesh import ShellMesh
    shell_mesh = ShellMesh('shell_mesh', design_geometry_class=[])