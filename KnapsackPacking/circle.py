import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, MultiPolygon
from ellipse import Ellipse

# resolution of the polygon approximating a circle used only for visualization; according to Shapely documentation, a resolution of 16 allows to cover 99.8% of the circle's area (https://shapely.readthedocs.io/en/stable/manual.html#object.buffer)
VISUALIZATION_RESOLUTION = 32

# resolution of the polygon approximating a circle used only for intersection calculations that need the actual intersection points, while non-approximating formula is used for the rest of checks (intersection/contains/within); 6-resolution gives 25 points
INTERSECTION_POINT_RESOLUTION = 6


class Circle(BaseGeometry):

    def __init__(self, center, radius):

        """Constructor"""

        self.center = center
        if type(self.center) != Point:
            self.center = Point(center[0], center[1])
        self.radius = radius

        # approximate polygon used only when specific intersection points with other shapes need to be found, while the real circle formulae are used for most calculus (including intersection/contains/within checks)
        self.polygon = self.center.buffer(self.radius, resolution=INTERSECTION_POINT_RESOLUTION)

    def __reduce__(self):
        return Circle, (self.center, self.radius)

    def intersects(self, other):

        """Returns True if geometries intersect, else False"""

        # two circles intersect if the distance between their center is no greater than the sum of their radii
        if type(other) == Circle:

            return self.center.distance(other.center) <= self.radius + other.radius

        # a circle intersects with a multi-polygon if it intersects with any of the composing polygons, either exterior or hole
        if type(other) == MultiPolygon:

            for geom in other.geoms:

                if self.intersects(geom):

                    return True

            return False

        # for ellipses, consider their polygon approximation
        if type(other) == Ellipse:
            other = other.polygon

        # a circle intersects with another shape if the distance from the shape to the center of the circle is not greater than the radius
        return other.distance(self.center) <= self.radius

    def within(self, other):

        """Returns True if geometry is within the other, else False"""

        # one circle is contained by another circle if the second one has greater radius and the center-to-center distance plus the sum of the radii is lesser than the diameter of the second circle
        if type(other) == Circle:

            return other.radius > self.radius and self.center.distance(other.center) + self.radius + other.radius < 2 * other.radius

        # for a circle to be inside a multi-polygon, it must be within the exterior polygon and not intersecting with the holes
        if type(other) == MultiPolygon:

            if self.within(other.geoms[0]):

                for geom in other.geoms:

                    for hole in geom.interiors:

                        if self.intersects(hole):

                            return False

                return True

            return False

        # for ellipses, consider their polygon approximation
        if type(other) == Ellipse:
            other = other.polygon

        # a circle is inside another shape if the center of the circle is contained in the shape and the minimum distance to the boundary is greater than the radius
        return other.contains(self.center) and other.exterior.distance(self.center) > self.radius

    def contains(self, other):

        """Returns True if the geometry contains the other, else False"""

        # for the circle-in-circle case, use the within implementation, in opposite order
        if type(other) == Circle:

            return other.within(self)

        # for multi-polygons, just it is just needed to check if the circle contains the boundary polygon
        elif type(other) == MultiPolygon:
            other = other.geoms[0]

        # for ellipses, consider their polygon approximation
        if type(other) == Ellipse:
            other = other.polygon

        # a circle contains another shape (e.g. polygon) if all the points of the shape are inside the circle, i.e. at a distance less than the radius
        for coord in other.exterior.coords:
            if self.center.distance(Point(coord)) >= self.radius:
                return False
        return True

    @property
    def area(self):

        """Unitless area of the geometry (float)"""

        return np.pi * self.radius * self.radius

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
