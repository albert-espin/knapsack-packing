import copy
import math
import random
import numpy as np
from shapely.geometry import LinearRing, MultiPolygon, Point, Polygon, MultiPoint, MultiLineString
from circle import Circle, VISUALIZATION_RESOLUTION
from ellipse import Ellipse


def get_bounds(shape):

    """Return a tuple with the (min_x, min_y, max_x, max_y) describing the bounding box of the shape"""

    # the circle bounds can be found with the center and radius
    if type(shape) == Circle:

        return shape.center.x - shape.radius, shape.center.y - shape.radius,  shape.center.x + shape.radius, shape.center.y + shape.radius

    # for an ellipse, use the approximate polygon
    if type(shape) == Ellipse:
        shape = shape.polygon

    return shape.bounds


def get_bounding_rectangle_center(shape):

    """Return the center of the bounding rectangle for the passed shape"""

    # for a circle, use the center
    if type(shape) == Circle:

        return shape.center.x, shape.center.y

    # for an ellipse, use the approximate polygon
    if type(shape) == Ellipse:
        shape = shape.polygon

    return (shape.bounds[0] + shape.bounds[2]) / 2, (shape.bounds[1] + shape.bounds[3]) / 2


def get_centroid(shape):

    """Return the centroid of a shape"""

    # for a circle, use the center
    if type(shape) == Circle:

        return shape.center

    # for an ellipse, use the approximate polygon
    if type(shape) == Ellipse:
        shape = shape.polygon

    return shape.centroid


def get_shape_exterior_points(shape, is_for_visualization=False):

    """Return the exterior points of a shape"""

    if type(shape) == LinearRing:

        return [coord[0] for coord in shape.coords], [coord[1] for coord in shape.coords]

    if type(shape) == MultiPolygon:

        return shape.geoms[0].exterior.xy

    # a circle has no finite exterior points, but for visualization a discrete set is generated
    if type(shape) == Circle:

        if is_for_visualization:

            return shape.center.buffer(shape.radius, VISUALIZATION_RESOLUTION).exterior.xy

        return None

    # for the ellipse, always use the approximate polygon
    if type(shape) == Ellipse:

        return shape.polygon.exterior.xy

    return shape.exterior.xy


def do_shapes_intersect(shape0, shape1):

    """Return whether the two passed shapes intersect with one another"""

    # non-native shape types need to be the ones calling intersection, to handle all cases
    if type(shape0) == Circle or type(shape0) == Ellipse:

        return shape0.intersects(shape1)

    elif type(shape1) == Circle or type(shape1) == Ellipse:

        return shape1.intersects(shape0)

    # default case for native-to-native shape test
    return shape0.intersects(shape1)


def get_intersection_points_between_shapes(shape0, shape1):

    """If the two passed shapes intersect in one or more points (a finite number) return all of them, otherwise return an empty list"""

    intersection_points = list()

    # the contour of polygons and multi-polygons is used for the check, to detect boundary intersection points
    if type(shape0) == Polygon:

        shape0 = shape0.exterior

    elif type(shape0) == MultiPolygon:

        shape0 = MultiLineString(shape0.boundary)

    if type(shape1) == Polygon:

        shape1 = shape1.exterior

    elif type(shape1) == MultiPolygon:

        shape1 = MultiLineString(shape1.boundary)

    # non-native shape types use their approximate polygon boundaries
    if type(shape0) == Circle or type(shape0) == Ellipse:

        shape0 = shape0.polygon.exterior

    if type(shape1) == Circle or type(shape1) == Ellipse:

        shape1 = shape1.polygon.exterior

    intersection_result = shape0.intersection(shape1)

    if intersection_result:

        if type(intersection_result) == Point:
            intersection_points.append((intersection_result.x, intersection_result.y))

        elif type(intersection_result) == MultiPoint:
            for point in intersection_result:
                intersection_points.append((point.x, point.y))

    return intersection_points


def does_shape_contain_other(container_shape, content_shape):

    """Return whether the first shape is a container of the second one, which in such case acts as the content of the first one"""

    # Shapely core shapes do not know about circle or ellipse, so let them handle the check
    if type(container_shape) == Circle or type(container_shape) == Ellipse:
        return container_shape.contains(content_shape)

    # use the standard within check
    return content_shape.within(container_shape)


def copy_shape(shape):

    """Create and return a deep copy of the passed shape"""

    if type(shape) == Circle:

        return Circle(shape.center, shape.radius)

    if type(shape) == Ellipse:

        return Ellipse(shape.center, shape.half_width, shape.half_height, shape.polygon)

    return copy.deepcopy(shape)


def get_rectangle_points_from_bounds(min_x, min_y, max_x, max_y):

    """Find and return as (x, y) tuples the rectangle points from the passed bounds"""

    return [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]


def create_equilateral_triangle(center, side):

    """Create an equilateral triangle (with conventional orientation) given its center and side length; following the method explained in: https://math.stackexchange.com/a/1344707"""

    root_of_3 = math.sqrt(3)
    vertex0 = (center[0], center[1] + root_of_3 / 3 * side)
    vertex1 = (center[0] - side * 0.5, center[1] - root_of_3 / 6 * side)
    vertex2 = (center[0] + side * 0.5, center[1] - root_of_3 / 6 * side)

    return Polygon([vertex0, vertex1, vertex2])


def create_square(center, side):

    """Create a square (with conventional orientation) given its center and side length"""

    half_side = side * 0.5
    return Polygon([(center[0] - half_side, center[1] - half_side), (center[0] + half_side, center[1] - half_side), (center[0] + half_side, center[1] + half_side), (center[0] - half_side, center[1] + half_side)])


def create_random_triangle_in_rectangle_corner(min_x, min_y, max_x, max_y):

    """Create a triangle that lies in a corner of the rectangle defined by the passed bounds, i.e. one of the vertices of the triangle is one of the points of the rectangle, and the other two vertices lie on sides of the rectangle"""

    # determine the points of the rectangle from its bounds
    rectangle_points = get_rectangle_points_from_bounds(min_x, min_y, max_x, max_y)

    # randomly select a point as one of the vertices of the triangle
    first_vertex = random.choice(rectangle_points)

    # the other two vertices have one (different for each of the two) of the coordinates in common with the first point, and the other two in the range of the bounds
    triangle_points = [first_vertex, (first_vertex[0], random.uniform(min_y, max_y)), (random.uniform(min_x, max_y), first_vertex[1])]

    # create the triangle
    return Polygon(triangle_points)


def create_random_quadrilateral_in_rectangle_corners(min_x, min_y, max_x, max_y):

    """Create a quadrilateral with a side in common with the rectangle defined by the passed bounds; two of the vertices of the quadrilateral coincide with two side-adjacent vertices of the rectangle, and the other two vertices lie on two parallel sides of the rectangle"""

    # first define two points, i.e. a segment, between the minimum and maximum value in one dimension, and random values (within the bounds) for the other dimension;     # then, add the vertices of one of two sides of the rectangle that are not cut by the segment
    has_horizontal_segment = bool(random.getrandbits(1))
    if has_horizontal_segment:
        quadrilateral_points = [(min_x, random.uniform(min_y, max_y)), (max_x, random.uniform(min_y, max_y))]
        if bool(random.getrandbits(1)):
            quadrilateral_points.extend([(max_x, max_y), (min_x, max_y)])
        else:
            quadrilateral_points.extend([(max_x, min_y), (min_x, min_y)])
    else:
        quadrilateral_points = [(random.uniform(min_x, max_x), min_y), (random.uniform(min_x, max_x), max_y)]
        if bool(random.getrandbits(1)):
            quadrilateral_points.extend([(min_x, max_y), (min_x, min_y)])
        else:
            quadrilateral_points.extend([(max_x, max_y), (max_x, min_y)])

    # create the quadrilateral
    return Polygon(quadrilateral_points)


def create_random_polygon(min_x, min_y, max_x, max_y, vertex_num):

    """Create a random polygon with the passed x and y bounds and the passed number of vertices; code adapted from: https://stackoverflow.com/a/45841790"""

    # generate the point coordinates within the bounds
    x = np.random.uniform(min_x, max_x, vertex_num)
    y = np.random.uniform(min_y, max_y, vertex_num)

    # determine the center of all points
    center = (sum(x) / vertex_num, sum(y) / vertex_num)

    # find the angle of each point from the center
    angles = np.arctan2(x - center[0], y - center[1])

    # sort points by their angle from the center to avoid self-intersections
    points_sorted_by_angle = sorted([(i, j, k) for i, j, k in zip(x, y, angles)], key=lambda t: t[2])

    # the process fails if there are duplicate points
    if len(points_sorted_by_angle) != len(set(points_sorted_by_angle)):
        return None

    # structure points as x-y tuples
    points = [(x, y) for (x, y, a) in points_sorted_by_angle]

    # create the polygon
    return Polygon(points)
