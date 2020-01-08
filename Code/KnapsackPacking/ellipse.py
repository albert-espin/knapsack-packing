import numpy as np
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import problem_solution

# resolution of the polygon approximating a circle then scaled to approximate the ellipsis; according to Shapely documentation, a resolution of 16 allows to cover 99.8% of the circle's area (https://shapely.readthedocs.io/en/stable/manual.html#object.buffer)
RESOLUTION = 16


class Ellipse(BaseGeometry):

    def __init__(self, center, half_width, half_height, polygon=None):

        """Constructor"""

        if type(center) != Point:
            center = Point(center[0], center[1])

        self.center = center
        self.half_width = half_width
        self.half_height = half_height

        # approximate the ellipse as a polygon to avoid having to define checks (intersection, is-within) with all other shapes
        circle_approximation = center.buffer(1, resolution=RESOLUTION)
        if not polygon:
            polygon = affinity.scale(circle_approximation, self.half_width, self.half_height)
        self.polygon = polygon

    def __reduce__(self):
        return Ellipse, (self.center, self.half_width, self.half_height, self.polygon)

    def intersects(self, other):

        """Returns True if geometries intersect, else False"""

        # use the approximate polygon for the check
        return problem_solution.do_shapes_intersect(self.polygon, other)

    def within(self, other):

        """Returns True if geometry is within the other, else False"""

        # use the approximate polygon for the check
        return problem_solution.does_shape_contain_other(other, self.polygon)

    def contains(self, other):

        """Returns True if the geometry contains the other, else False"""

        # use the approximate polygon for the check
        return problem_solution.does_shape_contain_other(self.polygon, other)

    @property
    def area(self):

        """Unitless area of the geometry (float)"""

        return np.pi * self.half_width * self.half_height

    @property
    def ctypes(self):
        return None

    @property
    def __array_interface__(self):
        return None

    def _set_coords(self, ob):
        return None

    @property
    def xy(self):
        return None

    @property
    def __geo_interface__(self):
        return None

    def svg(self, scale_factor=1., **kwargs):
        return None
