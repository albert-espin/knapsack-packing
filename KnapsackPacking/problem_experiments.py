import os
import pickle
import time
import numpy as np
from math import ceil
from multiprocessing import Pool
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
import evolutionary
import greedy
import reversible
import shape_functions
from circle import Circle
from common_algorithm_functions import get_time_since, visualize_boxplot_for_data_sequence, print_if_allowed, \
    visualize_plot, visualize_bar_plot, add_newlines_by_spaces, get_stats
from ellipse import Ellipse
from problem_solution import Item, Container, Problem, Solution

# types of problems that can be solved: Knapsack-Packing Joint Problem, or Packing Problem
KNAPSACK_PACKING_PROBLEM_TYPE = "KnapsackPacking"
PACKING_PROBLEM_TYPE = "Packing"

# directory where to save figures and results of the problems created specifically for the Knapsack-Packing Joint Problem
KNAPSACK_PACKING_PROBLEM_DIR = "../Output/Problems/CustomKnapsackPacking/Comparison/"

# directory where to save figures and results of instances of the Packing Problem
PACKING_PROBLEM_DIR = "../Output/Problems/Packing/Comparison/"


def create_knapsack_packing_problems_with_manual_solutions(can_print=False):

    """Create a set of Knapsack-Packing problem instances that are solved (optimally) with manual placements (using actions available for all the algorithms); both the problems and solutions are returned"""

    problems, solutions = list(), list()

    start_time = time.time()

    # Problem 1

    max_weight = 120.
    container_shape = Circle((3.3, 3.3), 3.3)
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0, 0), (0, 4.5), (4.5, 4.5), (4.5, 0)]), 40., 50.),
             Item(Circle((0, 0), 0.45), 20., 5.),
             Item(Circle((0, 0), 0.45), 20., 10.),
             Item(Circle((0, 0), 0.45), 20., 15.),
             Item(Circle((0, 0), 0.45), 20., 20.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (3.3, 3.3), 0.), can_print)
    print_if_allowed(solution.add_item(1, (3.3, 6.05), 0.), can_print)
    print_if_allowed(solution.add_item(2, (3.3, 0.55), 0.), can_print)
    print_if_allowed(solution.add_item(3, (6.05, 3.3), 0.), can_print)
    print_if_allowed(solution.add_item(4, (0.55, 3.3), 0.), can_print)

    # Problem 2

    max_weight = 100.
    container_shape = Point(5, 5).buffer(5, 4)
    container = Container(max_weight, container_shape)
    items = [Item(MultiPolygon([(Point(5, 5).buffer(4.7, 4).exterior.coords,
                                 [tuple(Point(5, 5).buffer(4, 4).exterior.coords)])]), 10., 25.),
             Item(MultiPolygon([(Point(5, 5).buffer(3.7, 4).exterior.coords,
                                 [tuple(Point(5, 5).buffer(3, 4).exterior.coords)])]), 10., 15.),
             Item(MultiPolygon([(Point(5, 5).buffer(2.7, 4).exterior.coords,
                                 [tuple(Point(5, 5).buffer(2, 4).exterior.coords)])]), 10., 20.),
             Item(MultiPolygon([(Point(5, 5).buffer(1.7, 4).exterior.coords,
                                 [tuple(Point(5, 5).buffer(1, 4).exterior.coords)])]), 20., 20.),
             Item(Circle((0., 0.), 0.7), 20., 10)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(1, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(2, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(3, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(4, (5., 5.), 0.), can_print)

    # Problem 3

    max_weight = 32.
    container_shape = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0, 0), (0, 6.), (6., 0)]), 10., 20.),
             Item(Polygon([(0, 0), (0, 6.), (6., 0)]), 10., 10.),
             Item(Ellipse((0, 0), 1.5, 0.3), 10., 5.),
             Item(Ellipse((0, 0), 3, 0.3), 5., 5.),
             Item(Ellipse((0, 0), 1.5, 0.3), 5., 5.),
             Item(Ellipse((0, 0), 3, 0.3), 10., 5.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (4.99, 5), 0.), can_print)
    print_if_allowed(solution.add_item(1, (5.01, 5), 180.), can_print)
    print_if_allowed(solution.add_item(3, (5., 1.65), 0), can_print)
    print_if_allowed(solution.add_item(4, (5., 8.35), 0), can_print)

    # Problem 4

    max_weight = 50.
    container_shape = Ellipse((3., 2.), 3., 2.)
    container = Container(max_weight, container_shape)
    items = [Item(Ellipse((0., 0.), 0.7, 0.5), 5., 7),
             Item(Ellipse((0., 0.), 0.3, 0.1), 7., 2),
             Item(Ellipse((0., 0.), 0.2, 0.4), 8., 4),
             Item(Ellipse((0., 0.), 0.5, 0.3), 3., 5),
             Item(Circle((0., 0.), 0.4), 4., 5),
             Item(Circle((0., 0.), 0.25), 3., 2),
             Item(Circle((0., 0.), 0.2), 9., 5),
             Item(Circle((0., 0.), 0.1), 4., 3.),
             Item(Circle((0., 0.), 0.7), 9., 3.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (3., 1.94), 0.), can_print)
    print_if_allowed(solution.add_item(2, (3., 3.24), 90.), can_print)
    print_if_allowed(solution.add_item(3, (3., 2.74), 0.), can_print)
    print_if_allowed(solution.add_item(4, (2.25, 3.5), 0.), can_print)
    print_if_allowed(solution.add_item(5, (3., 3.71), 0.), can_print)
    print_if_allowed(solution.add_item(6, (3.46, 3.75), 0.), can_print)
    print_if_allowed(solution.add_item(7, (3.44, 3.43), 0.), can_print)
    print_if_allowed(solution.add_item(8, (3., 0.72), 0.), can_print)

    # Problem 5

    max_weight = 100.
    container_shape = MultiPolygon([(((0, 0), (0.5, 3), (0, 5), (5, 4.5), (5, 0)),
                                     [((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)),
                                      ((0.3, 0.3), (0.3, 1.2), (1.6, 2.9), (0.75, 0.4)),
                                      ((3.1, 1.5), (3.5, 4.5), (4.9, 4.4), (4.8, 1.2))])])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0, 0), (1, 1), (1, 0)]), 15., 32.),
             Item(Polygon([(1, 2), (1.5, 3), (4, 5), (1, 4)]), 30., 100.),
             Item(MultiPolygon([(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                                 [((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)),
                                  ((0.3, 0.3), (0.3, 0.6), (0.6, 0.6), (0.6, 0.4))])]), 12., 30.),
             Item(Polygon([(0.1, 0.1), (0.1, 0.2), (0.2, 0.2)]), 10., 10.),
             Item(MultiPolygon([(((0., 0.), (0., 1.4), (2., 1.3), (2., 0.)),
                                 [((0.1, 0.1), (0.1, 0.15), (0.15, 0.15), (0.15, 0.1)),
                                  ((0.2, 0.2), (0.2, 1.2), (1.8, 1.1), (1.8, 0.2))
                                  ])]), 1., 5.),
             Item(Circle((0., 0.), 0.4), 1., 14.),
             Item(Circle((0., 0.), 0.1), 2., 12.),
             Item(Ellipse((0., 0.), 0.5, 0.2), 3., 12.),
             Item(Polygon([(0., 0.), (0., 0.3), (0.3, 0.3)]), 1., 10.),
             Item(Ellipse((0., 0.), 0.8, 0.3), 10., 12.),
             Item(Ellipse((0., 0.), 0.1, 0.05), 1., 2.),
             # random items
             # Item(shape_functions.create_random_polygon(0, 0, 0.8, 0.8, 10), 1., 5.),
             # Item(shape_functions.create_random_triangle_in_rectangle_corner(0, 0, 0.8, 0.8), 1., 5.),
             # Item(shape_functions.create_random_quadrilateral_in_rectangle_corners(0, 0, 0.8, 0.8), 1., 5.),
             # out-items
             Item(Circle((0., 0.), 0.2), 50., 1.),
             ]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    # solution.visualize()

    # print(solution.add_item(0, (1.2, 0.5), 150.))
    print_if_allowed(solution.add_item(0, (1.2, 0.5), 0.), can_print)
    print_if_allowed(solution.add_item(1, (2., 3.), 0.), can_print)
    print_if_allowed(solution.add_item(2, (2.5, 2.5), 0.), can_print)
    print_if_allowed(solution.move_item_in_direction(2, (1, -1), evolutionary.MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM, evolutionary.MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION, 9999), can_print)
    print_if_allowed(solution.add_item(3, (2.5, 2.4), 0.), can_print)
    print_if_allowed(solution.add_item(4, (3., 0.7), 0.), can_print)
    print_if_allowed(solution.add_item(5, (3.03, 0.73), 0.), can_print)
    print_if_allowed(solution.add_item(6, (3.45, 1.02), 0.), can_print)
    print_if_allowed(solution.add_item(7, (3., 3.82), 45.), can_print)
    print_if_allowed(solution.add_item(8, (2.4, 0.7), 0.), can_print)
    print_if_allowed(solution.move_item(0, (0.29, 0)), can_print)
    # print_if_allowed(solution.move_item_to(0, (1.49, 2.5)), can_print)
    print_if_allowed(solution.rotate_item_to(0, 180.), can_print)
    print_if_allowed(solution.rotate_item(0, 90.), can_print)
    # print_if_allowed(solution.remove_item(0), can_print)
    # print_if_allowed(solution.rotate_item(4, 20), can_print)
    # print_if_allowed(solution.move_item(7, (1, 0)), can_print)
    # print_if_allowed(solution.rotate_item(7, -45), can_print)
    # print_if_allowed(solution.move_item(5, (-0.4, 0)), can_print)
    print_if_allowed(solution.add_item(9, (1.2, 4.07), 15.), can_print)
    print_if_allowed(solution.add_item(10, (3.6, 0.45), 30.), can_print)
    # print_if_allowed(solution.add_item(11, (4.5, 0.5), 0.), can_print)

    # Problem 6

    max_weight = 150.
    container_shape = MultiPolygon([(((0., 0.), (5., 0.), (5., 5.), (0., 5.)),
                                     [((0.7, 0.7), (1.5, 0.7), (1.5, 1.5), (0.7, 1.5)),
                                      ((2.4, 0.3), (4.3, 0.3), (4.3, 4.3), (2.4, 4.3)),
                                      ((0.7, 2.7), (1.5, 2.7), (1.5, 3.5), (0.7, 3.5))])])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0., 0.), (1.6, 0.), (1.4, 0.2), (1.7, 1.)]), 6., 13.),
             Item(Polygon([(0., 0.), (1.6, 3.), (2.8, 2.9), (1.5, 2.7), (1.9, 1.6)]), 11., 12.),
             Item(Polygon([(0., 0.), (1.8, 1.5), (0., 2.8)]), 15., 25.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.2), (0., 0.2)]), 14., 10.),
             Item(Polygon([(0., 0.), (2.5, 0.), (1.5, 0.2), (0., 0.2)]), 10., 12.),
             Item(Polygon([(0., 0.), (1.6, 0.), (0.8, 0.45), (0.6, 0.7), (0., 0.45)]), 17., 8.),
             Item(Polygon([(0., 0.), (1.5, 0.), (0.8, 0.15), (0., 0.1)]), 13., 12.),
             Item(Polygon([(0., 0.), (1.5, 0.), (0.8, 0.15), (0., 0.1)]), 15., 7.),
             Item(Ellipse((0., 0.), 0.5, 0.3), 15., 8.),
             Item(Ellipse((0., 0.), 0.2, 0.8), 14., 21.),
             Item(Circle((0., 0.), 0.2), 18., 18.),
             Item(Circle((0., 0.), 0.6), 11., 12.),
             Item(Circle((0., 0.), 0.35), 12., 9.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (0.9, 2.02), 0.), can_print)
    print_if_allowed(solution.add_item(3, (0.78, 0.12), 0.), can_print)
    print_if_allowed(solution.add_item(4, (2.8, 0.12), 0.), can_print)
    print_if_allowed(solution.add_item(5, (0.8, 3.85), 0.), can_print)
    print_if_allowed(solution.add_item(6, (0.78, 0.3), 0.), can_print)
    print_if_allowed(solution.add_item(7, (2.3, 2.57), 90.), can_print)
    print_if_allowed(solution.add_item(8, (0.3, 2.98), 90.), can_print)
    print_if_allowed(solution.add_item(9, (2.17, 1.05), 0.), can_print)
    print_if_allowed(solution.add_item(10, (1.8, 0.45), 0.), can_print)
    print_if_allowed(solution.add_item(11, (1.77, 4.38), 0.), can_print)
    print_if_allowed(solution.add_item(12, (0.35, 4.63), 0.), can_print)

    # Problem 7

    max_weight = 122.
    container_shape = Polygon([(3.5, 0.6), (0.5, 0.9), (3.7, 5.5), (1.7, 4.), (0., 6.5), (0.2, 8.6), (0.8, 9.8), (1.7, 8.9), (2, 9.1), (4.4, 9.3), (4.2, 6.7), (4.9, 7.5), (6.5, 8.4), (6.6, 7.9), (7.4, 8.2), (8.7, 5.5), (9.3, 4.8), (6.3, 0.2), (5., 3.5), (5, 0.7), (3.5, 0.6)])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0, 3), (0, 2.), (4., 0)]), 5., 6.),
             Item(Polygon([(0, 0), (1., 2.), (2.5, 2), (1, 1.2)]), 10., 7.),
             Item(Polygon([(0, 1), (1, 2.), (3., 0)]), 9., 4.),
             Item(Polygon([(0, 0.5), (1, 1.), (3, 1), (2., 0)]), 19., 14.),
             Item(Polygon([(0, 0.6), (2, 1), (2., 1.5), (1.2, 1.5)]), 19., 15.),
             Item(Polygon([(0, 0), (0, 2.), (0.5, 2), (0.5, 0.5), (2.5, 0.5), (2.5, 0)]), 7., 15.),
             Item(MultiPolygon([(((0.0, 0.0), (0.0, 1.8), (1.0, 2.7), (2.3, 0.0)),
                                 [((0.2, 0.2), (0.2, 1.4), (0.7, 2.1), (1.8, 0.5))])]), 12., 6.),
             Item(MultiPolygon([(((0.0, 0.0), (1.0, 1.8), (2.0, 2.5), (2.6, 0.7)),
                                 [((0.2, 0.2), (1.2, 1.4), (2.1, 1.7))])]), 7., 13.),
             Item(Ellipse((0, 0), 0.5, 0.2), 4., 9.),
             Item(Ellipse((0, 0), 0.2, 1.5), 21., 14.),
             Item(Ellipse((0, 0), 2.5, 3.5), 16., 30.),
             Item(Circle((0, 0), 0.4), 7., 12.),
             Item(Circle((0, 0), 0.3), 10., 3.),
             Item(Circle((0, 0), 1.), 1., 3.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5.73, 3.02), 318.), can_print)
    print_if_allowed(solution.add_item(1, (6.3, 4.1), 40.), can_print)
    print_if_allowed(solution.add_item(2, (4.58, 2.5), 315.), can_print)
    print_if_allowed(solution.add_item(3, (1.3, 5.4), 320.), can_print)
    print_if_allowed(solution.add_item(4, (1.4, 1.7), 20.), can_print)
    print_if_allowed(solution.add_item(5, (2.9, 7.9), 180.), can_print)
    print_if_allowed(solution.add_item(6, (8.2, 4), 300.), can_print)
    print_if_allowed(solution.add_item(7, (2.5, 7.4), 340.), can_print)
    print_if_allowed(solution.add_item(8, (7.3, 4.), 320.), can_print)
    print_if_allowed(solution.add_item(9, (2.9, 3.9), 330.), can_print)
    print_if_allowed(solution.add_item(11, (7.8, 4.4), 0.), can_print)
    print_if_allowed(solution.add_item(13, (6.2, 6.8), 0.), can_print)

    # Problem 8

    max_weight = 100.
    container_shape = Polygon([(0., 0.), (0., 5.), (2.5, 3.4), (5., 5.), (5., 0), (2.5, 1.6)])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0., 0.), (0., 3.), (0.25, 3.), (0.25, 0.25), (2., 2.5), (3.75, 0.25), (3.75, 3.), (4., 3.), (4., 0.), (3.75, 0.), (2., 2.), (0.25, 0.)]), 100., 100.),
             Item(Polygon([(0., 0.), (1.6, 1.), (1.8, 1.9), (0.9, 1.6)]), 11., 12.),
             Item(Polygon([(0., 0.), (1.8, 2.5), (0., 1.8)]), 15., 5.),
             Item(Polygon([(0., 0.), (0.5, 0.), (1.2, 0.4), (0., 0.5)]), 4., 10.),
             Item(Polygon([(0., 0.), (2.5, 0.), (1.5, 0.2), (0., 0.5)]), 1., 2.),
             Item(Polygon([(0., 0.), (0.7, 0.25), (1.6, 1.5), (0.6, 0.7), (0., 0.45)]), 17., 8.),
             Item(Polygon([(0., 0.), (0.8, 0.5), (1.5, 1.2), (0., 0.5)]), 13., 11.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.2, 0.6), (0., 0.3)]), 15., 7.),
             Item(Ellipse((0., 0.), 0.6, 0.4), 15., 8.),
             Item(Ellipse((0., 0.), 2., 0.5), 15., 8.),
             Item(Ellipse((0., 0.), 0.5, 0.3), 24., 6.),
             Item(Ellipse((0., 0.), 0.4, 0.1), 4., 3.),
             Item(Circle((0., 0.), 0.6), 11., 2.),
             Item(Circle((0., 0.), 0.35), 12., 4.),
             Item(Circle((0., 0.), 0.2), 18., 8.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (2.5, 2.02), 0.), can_print)

    # Problem 9

    max_weight = 200.
    container_shape = Point(5, 5).buffer(5, 3)
    container = Container(max_weight, container_shape)
    items = [Item(MultiPolygon([(Point(5, 5).buffer(4.7, 2).exterior.coords,
                                 [((9., 5.), (5., 1.), (1., 5.), (5., 9.))])]), 120., 110.),
             Item(Polygon([(0., 0.), (0., 5.), (5., 5.), (5., 0.)]), 50., 80.),
             Item(Polygon([(1., 4.2), (1.5, 2.), (4., 0)]), 15., 14.),
             Item(Polygon([(0, 0), (1., 2.), (2.5, 2), (1, 1.2)]), 11., 11.),
             Item(Polygon([(0, 1), (1, 2.), (3., 0)]), 11., 4.),
             Item(Polygon([(0, 0.5), (1, 1.), (3, 1), (2., 0)]), 19., 14.),
             Item(Polygon([(0, 0.4), (1.8, .8), (1.5, 1.3), (1.2, 3.3)]), 17., 15.),
             Item(Polygon([(0, 0), (0, 2.), (0.9, 2), (0.9, 0.5), (1.5, 0.5), (1.5, 0)]), 70., 15.),
             Item(Ellipse((0, 0), 0.8, 1.2), 14., 13.),
             Item(Ellipse((0, 0), 1.2, 1.5), 12., 6.),
             Item(Ellipse((0, 0), 2.5, 1.7), 16., 10.),
             Item(Circle((0, 0), 0.7), 17., 11.),
             Item(Circle((0, 0), 0.8), 13., 10.),
             Item(Circle((0, 0), 1.), 4., 4.),
             Item(Circle((0, 0), 2.), 22., 8.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(1, (5., 5.), 45.), can_print)

    # Problem 10

    max_weight = 150.
    container_shape = Polygon([(2., 5.), (3., 5), (3., 3.), (5., 3.), (5., 2.), (3., 2.), (3., 0.), (2., 0.), (2., 2.), (0., 2.), (0., 3.), (2., 3.)])
    container = Container(max_weight, container_shape)
    items = [Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.45), (0., 0.45)]), 20., 30.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.45), (0., 0.45)]), 20., 30.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.1), (0., 0.1)]), 5., 25.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.1), (0., 0.1)]), 5., 25.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (4.23, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(1, (4.23, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(2, (0.77, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(3, (0.77, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(4, (2.48, 0.76), 270.), can_print)
    print_if_allowed(solution.add_item(5, (2.52, 0.76), 90.), can_print)
    print_if_allowed(solution.add_item(6, (2.48, 4.24), 270.), can_print)
    print_if_allowed(solution.add_item(7, (2.52, 4.24), 90.), can_print)
    print_if_allowed(solution.add_item(8, (2.5, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(9, (2.5, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(16, (2.5, 3.25), 0.), can_print)
    print_if_allowed(solution.add_item(17, (2.5, 1.75), 0.), can_print)
    print_if_allowed(solution.add_item(18, (1.64, 2.5), 90.), can_print)
    print_if_allowed(solution.add_item(19, (3.36, 2.5), 90.), can_print)

    # show elapsed time
    elapsed_time = get_time_since(start_time)
    print_if_allowed("Manual elapsed time: {} ms".format(round(elapsed_time, 3)), can_print)

    return problems, [str(i + 1) for i in range(len(problems))], solutions


def create_packing_problems_with_optimal_solution_values():

    """Create a set of Packing problem instances, returned along with the externally calculated optimal solution values (not the full solutions with their placements)"""

    problems, problem_names, optimal_values = list(), list(), list()

    # the capacity is set to infinite so that it never restricts placements; all items have value 1 so that the objective is to maximize the number of placed items
    max_weight = np.inf

    # Circles in circle; Wolfram Alpha query: "pack 7 circles of radius 3.9 in a circle of radius 13"; full link: https://www.wolframalpha.com/input/?i=pack+7+circles+of+radius+3.9+in+a+circle+of+radius+13
    container_shape = Circle((13, 13), 13)
    container = Container(max_weight, container_shape)
    item_num = 7
    items = [Item(Circle((0, 0), 3.9), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Circles in circle")
    optimal_values.append(item_num)

    # Triangles in circle; Wolfram Alpha query: "pack 20 triangles of side 4 in a circle of radius 9.5"; full link: https://www.wolframalpha.com/input/?i=pack+20+triangles+of+side+4+in+a+circle+of+radius+9.5
    container_shape = Circle((9.5, 9.5), 9.5)
    container = Container(max_weight, container_shape)
    item_num = 20
    items = [Item(shape_functions.create_equilateral_triangle((0, 0), 4), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Triangles in circle")
    optimal_values.append(item_num)

    # Squares in circle; Wolfram Alpha query: "pack 12 squares of side 3 in a circle of radius 7.8"; full link: https://www.wolframalpha.com/input/?i=pack+12+squares+of+side+3+in+a+circle+of+radius+7.8
    container_shape = Circle((7.8, 7.8), 7.8)
    container = Container(max_weight, container_shape)
    item_num = 12
    items = [Item(shape_functions.create_square((0, 0), 3), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Squares in circle")
    optimal_values.append(item_num)

    # Circles in triangle; Wolfram Alpha query: "pack 10 circles of radius 3 in a triangle of side 33"; full link: https://www.wolframalpha.com/input/?i=pack+10+circles+of+radius+3+in+a+triangle+of+side+33
    container_shape = shape_functions.create_equilateral_triangle((19, 9.5), 33)
    container = Container(max_weight, container_shape)
    item_num = 10
    items = [Item(Circle((0, 0), 3), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Circles in triangle")
    optimal_values.append(item_num)

    # Triangles in triangle; Wolfram Alpha query: "pack 18 triangles of side 3.5 in a triangle of side 20"; full link: https://www.wolframalpha.com/input/?i=pack+18+triangles+of+side+3.5+in+a+triangle+of+side+20
    container_shape = shape_functions.create_equilateral_triangle((12, 6), 20)
    container = Container(max_weight, container_shape)
    item_num = 18
    items = [Item(shape_functions.create_equilateral_triangle((0, 0), 3.5), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Triangles in triangle")
    optimal_values.append(item_num)

    # Squares in triangle; Wolfram Alpha query: "pack 30 squares of side 7.5 in a triangle of side 80"; full link: https://www.wolframalpha.com/input/?i=pack+24+squares+of+side+7.5+in+a+triangle+of+side+80
    container_shape = shape_functions.create_equilateral_triangle((49, 24.5), 80)
    container = Container(max_weight, container_shape)
    item_num = 30
    items = [Item(shape_functions.create_square((0, 0), 7.5), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Squares in triangle")
    optimal_values.append(item_num)

    # Circles in square; Wolfram Alpha query: "pack 50 circles of radius 17 in a square of side 300"; full link https://www.wolframalpha.com/input/?i=pack+50+circles+of+radius+17+in+a+square+of+side+300:
    container_shape = shape_functions.create_square((150, 150), 300)
    container = Container(max_weight, container_shape)
    item_num = 50
    items = [Item(Circle((0, 0), 17), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Circles in square")
    optimal_values.append(item_num)

    # Triangles in square; Wolfram Alpha query: "pack 15 triangles of side 4 in a square of side 14"; full link: https://www.wolframalpha.com/input/?i=pack+15+triangles+of+side+4+in+a+square+of+side+14
    container_shape = shape_functions.create_square((7, 7), 14)
    container = Container(max_weight, container_shape)
    item_num = 15
    items = [Item(shape_functions.create_equilateral_triangle((0, 0), 4), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Triangles in square")
    optimal_values.append(item_num)

    # Squares in square; Wolfram Alpha query: "pack 100 squares of side 4 in a square of side 58"; full link: https://www.wolframalpha.com/input/?i=pack+100+squares+of+side+4+in+a+square+of+side+58
    container_shape = shape_functions.create_square((22.5, 22.5), 58)
    container = Container(max_weight, container_shape)
    item_num = 100
    items = [Item(shape_functions.create_square((0, 0), 4), 1., 1.)] * item_num
    problem = Problem(container, items)
    problems.append(problem)
    problem_names.append("Squares in square")
    optimal_values.append(item_num)

    return problems, problem_names, optimal_values


def create_problems(problem_type, can_print=False):

    """Create a set of problems of the specified problem type; both the problems and their optimal solutions (or at least optimal solution values) are returned"""

    if problem_type == KNAPSACK_PACKING_PROBLEM_TYPE:
        return create_knapsack_packing_problems_with_manual_solutions(can_print)

    elif problem_type == PACKING_PROBLEM_TYPE:
        return create_packing_problems_with_optimal_solution_values()

    return None, None, None


def execute_algorithm_with_params(params):

    """Execute the algorithm specified in the first of the passed parameters with the rest of parameters, and return the solution, value and elapsed time"""

    # unpack the algorithm and its parameters
    algorithm, algorithm_name, problem, show_solution_plot, solution_plot_save_path, calculate_times, calculate_value_evolution = params

    start_time = time.time()
    value_evolution = None
    times_dict = None
    if calculate_value_evolution:
        if algorithm == evolutionary.solve_problem:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(problem, calculate_times=calculate_times, return_population_fitness_per_generation=calculate_value_evolution)
            else:
                solution, value_evolution = algorithm(problem, calculate_times=calculate_times, return_population_fitness_per_generation=calculate_value_evolution)
        else:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(problem, calculate_times=calculate_times, return_value_evolution=calculate_value_evolution)
            else:
                solution, value_evolution = algorithm(problem, calculate_times=calculate_times, return_value_evolution=calculate_value_evolution)
    elif calculate_times:
        solution, times_dict = algorithm(problem, calculate_times=calculate_times)
    else:
        solution = algorithm(problem)
    elapsed_time = get_time_since(start_time)

    if solution and (show_solution_plot or solution_plot_save_path):
        solution.visualize(show_plot=show_solution_plot, save_path=solution_plot_save_path)

    return solution, solution.value, value_evolution, elapsed_time, times_dict


def execute_algorithm(algorithm, algorithm_name, problem, show_solution_plot=False, solution_plot_save_path=None, calculate_times=False, calculate_fitness_stats=False, execution_num=1, process_num=1):

    """Execute the passed algorithm as many times as specified (with each execution in a different CPU process if indicated), returning (at least) lists with the obtained solutions, values and elapsed times (one per execution)"""

    # encapsulate the algorithm and its parameters in a tuple for each execution (needed for multi-processing)
    param_tuples = [(algorithm, algorithm_name, problem, show_solution_plot, solution_plot_save_path, calculate_times, calculate_fitness_stats) for _ in range(execution_num)]

    solutions, values, value_evolutions, times, time_divisions = list(), list(), list(), list(), list()

    # if possible, perform each execution in a separate CPU process (in parallel)
    if process_num > 1:

        process_pool = Pool(process_num)
        batch_num = ceil(execution_num / process_num)
        for batch in range(batch_num):
            results = process_pool.map(execute_algorithm_with_params, param_tuples[batch * process_num: batch * process_num + process_num])
            batch_solutions, batch_values, batch_value_evolutions, batch_times, batch_time_divisions = [result[0] for result in results], [result[1] for result in results], [result[2] for result in results], [result[3] for result in results], [result[4] for result in results]
            solutions.extend(batch_solutions)
            values.extend(batch_values)
            value_evolutions.extend(batch_value_evolutions)
            times.extend(batch_times)
            time_divisions.extend(batch_time_divisions)
            '''process_pool.terminate()
            process_pool.join()'''

    # perform the calculation sequentially if multi-processing is not allowed
    else:

        for i in range(execution_num):

            solution, value, value_evolution, elapsed_time, time_division = execute_algorithm_with_params(param_tuples[i])
            solutions.append(solution)
            values.append(value)
            value_evolutions.append(value_evolution)
            times.append(elapsed_time)
            time_divisions.append(time_division)

    return solutions, values, value_evolutions, times, time_divisions


def perform_experiments(problem_type, output_dir, load_experiments):

    """Perform a set of experiments for the problem with the passed index, and producing output in the specified directory (when applicable)"""

    experiment_file_path = output_dir + "experiments.pickle"

    # data structure where to store all the problems, and the experimental results for each algorithm: solutions, final values and times
    experiment_dict = dict()

    # if experiments should be loaded (not repeated), do it if possible
    if load_experiments:
        with open(experiment_file_path, "rb") as _file:
            experiment_dict = pickle.load(_file)

    # perform experiments if pre-existing results were not loaded
    if not experiment_dict:

        # given a problem type, create a set of problem instances that are solved (optimally) with manual placements (using actions available for all the algorithms)
        problems, problem_names, manual_solutions = create_problems(problem_type)

        if problems and problem_names and manual_solutions:

            # parameters for the experimentation; note: calculating internal times and value evolution can increase the overall time of algorithms (in a slight, almost neglectible way)
            execution_num = 10  # 1
            process_num = 10  # 1
            calculate_internal_times = True
            calculate_value_evolution = True

            start_time = time.time()

            # solve each problem with each algorithm
            for i, (problem, problem_name, solution) in enumerate(zip(problems, problem_names, manual_solutions)):

                experiment_dict[problem_name] = {"problem": problem, "manual_solution": solution, "algorithms": dict()}

                # solve the problem with different algorithms, executing each one multiple times to gain statistical significance
                for (algorithm_name, algorithm) in [("Greedy", greedy.solve_problem), ("Reversible", reversible.solve_problem), ("Evolutionary", evolutionary.solve_problem)]:

                    solutions, values, value_evolutions, times, time_divisions = execute_algorithm(algorithm=algorithm, algorithm_name=algorithm_name, problem=problem, execution_num=execution_num, process_num=process_num, calculate_times=calculate_internal_times, calculate_fitness_stats=calculate_value_evolution)
                    experiment_dict[problem_name]["algorithms"][algorithm_name] = {"solutions": solutions, "values": values, "value_evolutions": value_evolutions, "times": times, "time_divisions": time_divisions}

            # show the total time spent doing experiments (note that significant overhead can be introduced beyond calculation time if plots are shown or saved to files; for strict time measurements, plotting should be avoided altogether)
            print("Total experimental calculation time: {} s".format(round(get_time_since(start_time) / 1000.), 3))

    if experiment_dict:

        # experiment-saving parameter
        save_experiments = True

        # if possible, save the experiments to a binary file
        if not load_experiments and save_experiments:
            with open(experiment_file_path, "wb") as file:
                pickle.dump(experiment_dict, file)

        # visualization/saving parameters
        show_problem_stats = False
        save_problem_stats = False  # True
        show_manual_solution_plots = False
        save_manual_solution_plots = False  # True
        show_algorithm_solution_plots = False
        save_algorithm_solution_plots = False  # True
        show_value_evolution_plots = False
        save_value_evolution_plots = False  # True
        show_time_division_plots = False
        save_time_division_plots = False  # True
        show_algorithm_comparison = False
        save_algorithm_comparison = False  # True
        show_aggregated_result_tables = True
        save_aggregated_result_tables = False  # True

        # show/save the results of the experiments
        visualize_and_save_experiments(experiment_dict, output_dir, can_plots_show_value_and_weight=problem_type == KNAPSACK_PACKING_PROBLEM_TYPE, show_problem_stats=show_problem_stats, save_problem_stats=save_problem_stats, show_manual_solution_plots=show_manual_solution_plots, save_manual_solution_plots=save_manual_solution_plots, show_algorithm_solution_plots=show_algorithm_solution_plots, save_algorithm_solution_plots=save_algorithm_solution_plots, show_value_evolution_plots=show_value_evolution_plots, save_value_evolution_plots=save_value_evolution_plots, show_time_division_plots=show_time_division_plots, save_time_division_plots=save_time_division_plots, show_algorithm_comparison=show_algorithm_comparison, save_algorithm_comparison=save_algorithm_comparison, show_aggregated_result_tables=show_aggregated_result_tables, save_aggregated_result_tables=save_aggregated_result_tables)

    else:
        print("The experiments cannot be performed (there are no problems available).")


def visualize_and_save_experiments(experiment_dict, output_dir, can_plots_show_value_and_weight=True, show_problem_stats=False, save_problem_stats=True, show_manual_solution_plots=False, save_manual_solution_plots=True, show_algorithm_solution_plots=False, save_algorithm_solution_plots=True, show_value_evolution_plots=False, save_value_evolution_plots=True, show_time_division_plots=False, save_time_division_plots=True, show_algorithm_comparison=False, save_algorithm_comparison=True, show_aggregated_result_tables=True, save_aggregated_result_tables=True):

    """Show and/or save different plots of the results of the experiments represented in the passed dictionary, as specified by the parameters"""

    # if needed, save statistics about the problems and their manual solutions
    if show_problem_stats or save_problem_stats:
        problem_names = list(experiment_dict.keys())
        fields = ["Item num.", "Opt. % item num. in cont.", "Opt. % item value in cont.", "Item weight % of max weight", "Item area % of max area", "Cont. weight satur. %", "Cont. area satur. %"]
        data_frame = pd.DataFrame(index=problem_names, columns=fields)
        for problem_name in experiment_dict.keys():
            problem = experiment_dict[problem_name]["problem"]
            solution = experiment_dict[problem_name]["manual_solution"]
            if type(solution) == Solution:
                problem_results = [
                    len(problem.items),
                    round(len(solution.placed_items) / len(problem.items) * 100, 2),
                    round(sum([problem.items[item_index].value for item_index in solution.placed_items.keys()]) / sum([item.value for item in problem.items.values()]) * 100, 2),
                    round(sum([item.weight for item in problem.items.values()]) / problem.container.max_weight * 100, 2),
                    round(sum([item.shape.area for item in problem.items.values()]) / problem.container.shape.area * 100, 2),
                    round(sum([problem.items[item_index].weight for item_index in solution.placed_items.keys()]) / problem.container.max_weight * 100, 2),
                    round(sum([problem.items[item_index].shape.area for item_index in solution.placed_items.keys()]) / problem.container.shape.area * 100, 2),
                ]
                data_frame.loc[problem_name] = problem_results
        if len(data_frame) > 0:
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            min_row = data_frame.min()
            max_row = data_frame.max()
            mean_row = data_frame.mean()
            std_row = data_frame.std()
            data_frame.loc["Min"] = round(min_row, 2)
            data_frame.loc["Max"] = round(max_row, 2)
            data_frame.loc["Mean"] = round(mean_row, 2)
            data_frame.loc["Std"] = round(std_row, 2)
            data_frame.loc["Std / (max - min) %"] = round(std_row / (max_row - min_row) * 100, 2)
            if show_problem_stats:
                print(data_frame.to_string())
            if save_problem_stats:
                data_frame.to_excel(output_dir + "problem_stats.xlsx")
                data_frame.to_latex(output_dir + "problem_stats.tex")

    # create the problem results directory (if not done yet)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # for each problem's results, show or save plots if needed
    if show_manual_solution_plots or save_manual_solution_plots or show_algorithm_solution_plots or save_algorithm_solution_plots or show_algorithm_comparison or save_algorithm_comparison:
        for problem_name in experiment_dict.keys():

            problem, manual_solution, algorithm_dict = experiment_dict[problem_name].values()

            # create a subdirectory to store the solutions of the problem (if not done yet)
            problem_dir_path = output_dir + problem_name + "/"
            if not os.path.exists(problem_dir_path):
                os.mkdir(problem_dir_path)

            plotted_problem_name = "Problem " + problem_name if len(problem_name) < 5 else problem_name

            # if needed, show/save a plot of the initial state (empty solution), and the final state of a manual solution
            if show_manual_solution_plots or save_manual_solution_plots:
                empty_solution = Solution(problem)
                empty_solution.visualize(title_override=plotted_problem_name + " - Initial state", show_plot=show_manual_solution_plots, save_path=problem_dir_path + "empty_solution.png" if save_manual_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)
                if type(manual_solution) == Solution:
                    manual_solution.visualize(title_override=plotted_problem_name + " - Manual solution", show_plot=show_manual_solution_plots, save_path=problem_dir_path + "manual_solution.png" if save_manual_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)

            # if required, show/save plots of the solutions of each algorithm
            if show_algorithm_solution_plots or save_algorithm_solution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, solution in enumerate(subdict["solutions"]):
                        solution.visualize(title_override=plotted_problem_name + " - " + algorithm_name + " solution", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_solution.png" if save_algorithm_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)

            # if required, show/save plots of the value evolution of each algorithm
            if show_value_evolution_plots or save_value_evolution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, value_evolution in enumerate(subdict["value_evolutions"]):
                        if value_evolution:
                            if type(value_evolution) == list and type(value_evolution[0]) == list:
                                visualize_boxplot_for_data_sequence(data_lists=value_evolution, title=plotted_problem_name + " - Population fitness per generation", show_plot=show_value_evolution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_fitness_evolution.png" if save_value_evolution_plots else None)
                            else:
                                visualize_plot(values=value_evolution, title=plotted_problem_name + " - " + algorithm_name + " solution value per iteration", show_plot=show_value_evolution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_value_evolution.png" if save_value_evolution_plots else None)

            # if required, show/save plots of the time division in tasks of each algorithm
            if show_time_division_plots or save_time_division_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    for i, time_division in enumerate(subdict["time_divisions"]):
                        visualize_bar_plot(values=[value_pair[0] for value_pair in time_division.values()], labels=[add_newlines_by_spaces(label, 7) for label in list(time_division.keys())], title="Problem " + problem_name + " - " + algorithm_name + " time per task (milliseconds)", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_time_division.png" if save_algorithm_solution_plots else None)

            # if needed, show/save plots that compare the value and time of each algorithm considering multiple executions
            if show_algorithm_comparison or save_algorithm_comparison:
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["values"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Solution value by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), show_plot=show_algorithm_comparison, save_path=problem_dir_path + "value_comparison.png" if save_algorithm_comparison else None)
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["times"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Computational time (milliseconds) by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), y_scale_override="log", show_plot=show_algorithm_comparison, save_path=problem_dir_path + "time_comparison.png" if save_algorithm_comparison else None)

    # if needed, save tables with an aggregation of the value and time results of the executions of each problem (or just show them)
    if show_aggregated_result_tables or save_aggregated_result_tables:
        problem_names = list(experiment_dict.keys())
        algorithm_names = [algo_name for algo_name in experiment_dict[problem_names[0]]["algorithms"].keys()]
        fields = ["mean", "std", "min", "med", "max"]
        for concept in ["value", "time"]:
            algo_field_tuples = [(algo_name, field) for algo_name in algorithm_names for field in fields]
            if concept == "value":
                algo_field_tuples += [("Manual", "optim.")]
            multi_index = pd.MultiIndex.from_tuples(algo_field_tuples, names=["Algorithm", "Statistic"])
            data_frame = pd.DataFrame(index=problem_names, columns=multi_index)
            for problem_name in experiment_dict.keys():
                problem_results = list()
                for algo_name in algorithm_names:
                    mean, std, min_, median, max_ = get_stats(experiment_dict[problem_name]["algorithms"][algo_name][concept + "s"], 2 if concept == "value" else 0)
                    problem_results.extend([mean, std, min_, median, max_])
                if concept == "value":
                    if type(experiment_dict[problem_name]["manual_solution"]) == Solution:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"].value)
                    else:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"])
                data_frame.loc[problem_name] = problem_results
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            if show_aggregated_result_tables:
                print("{} results:\n{}\n".format(concept.capitalize(), data_frame.to_string()))
            if save_aggregated_result_tables:
                data_frame.to_excel(output_dir + concept + "_results.xlsx")
                data_frame.to_latex(output_dir + concept + "_results.tex")


def main():

    """Main function"""

    # set the type of problem to solve and the output directory
    # problem_type, output_dir = KNAPSACK_PACKING_PROBLEM_TYPE, KNAPSACK_PACKING_PROBLEM_DIR
    problem_type, output_dir = PACKING_PROBLEM_TYPE, PACKING_PROBLEM_DIR

    # whether it should be attempted to load existing experiments, and avoid running new ones
    load_experiments = True

    # perform (or just load and show) a set of experiments
    perform_experiments(problem_type, output_dir, load_experiments)


if __name__ == "__main__":
    main()
