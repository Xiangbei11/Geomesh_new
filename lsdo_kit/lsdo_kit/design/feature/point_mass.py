import numpy as np

class PointMass(object):
    def __init__(self, name, pointset):
        super().__init__(name, pointset)

    def add_mass(self, mass):
        if len(mass) == 1:
            self.mass = mass

            