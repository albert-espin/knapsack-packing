import copy
import itertools
import math
import random
import time
import numpy as np
import greedy
from circle import Circle
from common_algorithm_functions import get_time_since
from ellipse import Ellipse
from shape_functions import get_bounds, create_random_polygon, do_shapes_intersect, \
    create_random_triangle_in_rectangle_corner, does_shape_contain_other, \
    create_random_quadrilateral_in_rectangle_corners

# size of the population
POPULATION_SIZE = 100

# size of the offspring population created in each generation
OFFSPRING_SIZE = POPULATION_SIZE * 2

# size of the elite population subset of the algorithm
ELITE_SIZE = 5

# size of the tournament pool used for parent selection
PARENT_SELECTION_POOL_SIZE = 3

# size of the tournament pool used for population update
POPULATION_UPDATE_POOL_SIZE = 3

# maximum number of iterations for the generation of an initial solution
INITIAL_SOLUTION_GENERATION_MAX_ITER_NUM = 300

# maximum number of iterations to assume that a generated initial solution has converged
INITIAL_SOLUTION_GENERATION_CONVERGE_ITER_NUM = 50

# minimum number of iterations of mutation
MUTATION_MIN_ITER_NUM = 5

# probability weight to add an item in a mutation step
MUTATION_ADD_WEIGHT = 0.6

# probability weight to remove an item in a mutation step
MUTATION_REMOVE_WEIGHT = 0.1

# probability weight to modify an item's placement in a mutation step
MUTATION_MODIFY_WEIGHT = 0.3

# maximum number of times that the addition of a specific item in a mutation step should be attempted
MUTATION_ADD_MAX_ATTEMPT_NUM = 10

# maximum number of times that the modification of a specific item's placement in a mutation step should be attempted
MUTATION_MODIFY_MAX_ATTEMPT_NUM = 3

# maximum proportion of the container's side length (for each axis) that can be applied as a position offset in a mutation step of placement modification of the type small position change
MUTATION_MODIFY_SMALL_POSITION_CHANGE_PROPORTION = 0.2

# maximum proportion of 360 degrees that can be applied as a rotation offset in a mutation step of placement modification of the type small rotation change
MUTATION_MODIFY_SMALL_ROTATION_CHANGE_PROPORTION = 0.2

# maximum number of positions to check in the placement modification mutation that tries to move an item in a direction until it intersects
MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM = 15

# proportion of the side length of the bounding rectangle of the container used to calculate the minimum relevant distance between the reference point of an item and the first intersection point determined by the direction of a move-until-intersection mutation.
MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION = 0.03

# maximum number of angles to check in the placement modification mutation that tries to rotate an item in a direction until it intersects
MUTATION_MODIFY_ROTATE_UNTIL_INTERSECTION_ANGLE_NUM = 8

# probability of selecting an intermediate-step solution as the mutation output if it is better than the solution resulting from the last step
MUTATION_INTERMEDIATE_SELECTION_PROB = 1

# whether crossover can be used
CAN_USE_CROSSOVER = False

# probability to ignore mutation of a crossover-generated solution if it was better (higher fitness) before mutation
CROSSOVER_IGNORE_MUTATION_PROBABILITY = 0.5

# maximum number of attempts to try to produce a partitioning shape for crossover that intersects with the container (covering part of its area)
CROSSOVER_MAX_ATTEMPT_NUM = 5

# minimum proportion of the side length of the container's bounding rectangle (for the relevant axis) that can be used to define the length (in the same axis) of a non-segment partitioning shape in crossover
CROSSOVER_SHAPE_MIN_LENGTH_PROPORTION = 0.25

# maximum proportion of the side length of the container's bounding rectangle (for the relevant axis) that can be used to define the length (in the same axis) of a non-segment partitioning shape in crossover
CROSSOVER_SHAPE_MAX_LENGTH_PROPORTION = 0.75

# maximum number of vertices that can be used to generate a polygon to partition the container space of parents in two regions when performing crossover
CROSSOVER_POLYGON_MAX_VERTEX_NUM = 10

# maximum number of permutations determining the possible order in which item that lied partially in both regions of a crossover partitioning can be tried to be placed again in the container
CROSSOVER_MAX_PERMUTATION_NUM = 5

# proportion of individuals in a population that the candidate solutions generated in crossover permutations (for items that lied in both regions of the partitioning of the container) need to outperform in fitness to be eligible in the next population update
CROSSOVER_MIN_FITNESS_FOR_NON_BEST_PROPORTION = 0.95

# minimum proportion of the container's area that a closed-region shape needs to have to be accepted as a valid partitioning shape in crossover
CROSSOVER_SHAPE_MIN_AREA_PROPORTION = 0.1

# maximum number of generations
MAX_GENERATION_NUM = 30

# maximum number of generations to assume that algorithm has converged
CONVERGE_GENERATION_NUM = 12

# maximum proportion of iterations of the generation of an initial solution in which to specialize in trying to place a specific item
INITIAL_SOLUTION_GENERATION_FIRST_ITEM_SPECIALIZATION_ITER_PROPORTION = 0.5


def get_fitness(solution):

    """Return the fitness value of the passed solution"""

    if not solution:

        return 0

    # the fitness of a valid solution coincides with its value
    return solution.value


def get_fittest_solution(solutions):

    """Return the fittest of the passed solutions, using tie-breaking criteria if there are ties and only using random tie-breaking as a last resort"""

    # ensure that there are solutions
    if not solutions:

        return None

    # a single solution is the fittest
    if len(solutions) == 1:

        return solutions[0]

    fittest_solutions = list()
    max_fitness = -np.inf

    # find the fittest solutions
    for solution in solutions:

        fitness = get_fitness(solution)

        if fitness > max_fitness:

            fittest_solutions = [solution]
            max_fitness = fitness

        elif fitness == max_fitness:

            fittest_solutions.append(solution)

    # there may be no fittest solution
    if not fittest_solutions:

        return None

    # if there is only one fittest solution, return that one
    if len(fittest_solutions) == 1:

        return fittest_solutions[0]

    # for ties, use minimum-area optimization as tie-break criterion
    min_area_solutions = list()
    min_area = np.inf
    for solution in fittest_solutions:

        area = solution.get_area()

        if area < min_area:

            min_area_solutions = [solution]
            min_area = area

        elif area == min_area:

            min_area_solutions.append(solution)

    # there may be no tie-broken solution; return a random one of the fittest
    if not min_area_solutions:

        return random.choice(fittest_solutions)

    # if there is only one tie-broken solution, return that one
    if len(min_area_solutions) == 1:

        return min_area_solutions[0]

    # if the tie persists, use minimum-global-bounding-rectangle-area optimization as tie-break criterion
    min_bound_area_solutions = list()
    min_bound_area = np.inf
    for solution in min_area_solutions:

        bound_area = solution.get_global_bounding_rectangle_area()

        if bound_area < min_bound_area:

            min_bound_area_solutions = [solution]
            min_bound_area = bound_area

        elif bound_area == min_bound_area:

            min_bound_area_solutions.append(solution)

    # there may be no tie-broken solution; return a random one of those with minimum area
    if not min_bound_area_solutions:

        return random.choice(min_area_solutions)

    # if there is only one tie-broken solution, return that one
    if len(min_bound_area_solutions) == 1:

        return min_bound_area_solutions[0]

    # if the tie persists, uses randomness among the final tied solutions
    return random.choice(min_bound_area_solutions)


def get_fittest_solutions(solutions, size, break_ties=True):

    """Return the fittest of the solutions, as many as indicated"""

    # if there are less solutions than wanted, return them all
    if len(solutions) <= size:
        return solutions

    # sort solutions by descending fitness
    solutions_by_fitness = sorted(solutions, key=lambda solution: get_fitness(solution), reverse=True)

    # just truncate (if needed) the fitness-sorted list of individuals if ties should not be broken
    if not break_ties:
        return solutions_by_fitness[:size]

    # if the last item within the size has a different fitness than the next one (first one left out), it is safe to truncate
    if len(solutions_by_fitness) == size or get_fitness(solutions_by_fitness[size - 1]) != get_fitness(solutions_by_fitness[size]):
        return solutions_by_fitness[:size]

    # otherwise, find the first and last index with the tied-at-end fitness
    tied_at_end_fitness = get_fitness(solutions_by_fitness[size])
    start_index = size - 1
    end_index = size
    for i in range(end_index, len(solutions_by_fitness)):
        if i != end_index:
            if get_fitness(solutions_by_fitness[i]) == tied_at_end_fitness:
                end_index = i
            else:
                break
    for i in range(start_index, 0, -1):
        if i != start_index:
            if get_fitness(solutions_by_fitness[i]) == tied_at_end_fitness:
                start_index = i
            else:
                break
    tied_at_end_solutions = solutions_by_fitness[start_index:end_index]
    solutions_by_fitness = solutions_by_fitness[:start_index]

    while len(solutions_by_fitness) != size:
        tie_breaker_solution = get_fittest_solution(tied_at_end_solutions)
        tied_at_end_solutions.remove(tie_breaker_solution)
        solutions_by_fitness.append(tie_breaker_solution)

    return solutions_by_fitness


def generate_initial_solution(problem, item_index_to_place_first=-1, item_specialization_iter_proportion=0., calculate_times=False):

    """Generate an initial solution for the passed problem trying to place randomly selected items until some termination criteria is met"""

    # use the greedy algorithm without weighting, with pure random choices
    return greedy.solve_problem(problem, greedy_score_function=greedy.get_constant_score, repetition_num=1, max_iter_num=INITIAL_SOLUTION_GENERATION_MAX_ITER_NUM, max_iter_num_without_changes=INITIAL_SOLUTION_GENERATION_CONVERGE_ITER_NUM, item_index_to_place_first=item_index_to_place_first, item_specialization_iter_proportion=item_specialization_iter_proportion, calculate_times=calculate_times)


def generate_population(problem, population_size, item_specialization_iter_proportion):

    """Generate a population of the passed size for the passed problem"""

    # find the items whose weight does not exceed the container's capacity
    feasible_item_indices = [index for (index, item) in problem.items.items() if item.weight <= problem.container.max_weight]

    # limit the initial number of items to the population size (so that each of the kept items can have at least one specialization solution), picking a random sample of them if exceeding it
    if len(feasible_item_indices) > population_size:
        random.shuffle(feasible_item_indices)
        feasible_item_indices = feasible_item_indices[:population_size]

    # to promote the diversification of solutions in the search space, each feasible item will have the same number of solutions where it is tried to be insistently placed first, before any other item, to promote that all items appear at least in some initial solutions
    solution_num_per_item_specialization = population_size // len(feasible_item_indices)

    population = list()

    # for each feasible item, initialize a certain number of solutions with that item placed first (if possible)
    for item_index in feasible_item_indices:

        population.extend([generate_initial_solution(problem, item_index, item_specialization_iter_proportion) for _ in range(solution_num_per_item_specialization)])

    # create as many solutions with standard initialization as needed to reach the wanted population size
    remaining_solution_num = population_size - len(population)
    population.extend([generate_initial_solution(problem) for _ in range(remaining_solution_num)])

    return population


def get_tournament_winner(population, pool_size, individual_to_ignore=None):

    """Select and return the individual that is the winner (in terms of fitness) among a randomly picked pool of individuals from the population"""

    # ignore an individual if needed
    if individual_to_ignore:
        population = population[:]
        population.remove(individual_to_ignore)

    # the actual pool size can be shrunk if the population is too small
    pool_size = min(pool_size, len(population))

    # if the pool size is 0, no winner can be found
    if pool_size == 0:
        return None

    # otherwise find the individuals of the pool, allowing repetition
    pool = [random.choice(population) for _ in range(pool_size)]

    # the fittest individual is the winner
    return get_fittest_solution(pool)


def select_parents(population, offspring_size, pool_size, can_use_crossover):

    """Select (with a tournament pool of the passed size) and return the parents among the passed population so that they can later generate the passed number of offspring"""

    # if crossover will be used, pairs of parents will be selected
    if can_use_crossover:
        selection_num = offspring_size // 2
        if offspring_size % 2 != 0:
            selection_num += 1

    # for non-crossover cases, just one offspring will be generated per parent, using mutation
    else:
        selection_num = offspring_size

    parents = list()

    # select the parents
    for i in range(selection_num):

        # for crossover, parents are represented as pairs, and they cannot be the same individual
        if can_use_crossover:
            parent0 = get_tournament_winner(population, pool_size)
            if parent0:
                parent1 = get_tournament_winner(population, pool_size, parent0)
                if parent1:
                    parents.append((parent0, parent1))

        # when there is no crossover, only mutation, parents are selected and represented individually
        else:
            parent = get_tournament_winner(population, pool_size)
            if parent:
                parents.append(parent)

    return parents


def get_crossover(parent0, parent1, max_attempt_num, shape_min_length_proportion, shape_max_length_proportion, shape_min_area_proportion, polygon_max_vertex_num, max_permutation_num, min_fitness_for_non_best):

    """Generate and return two offspring from the passed parents"""

    # determine the bounds and side length of the container (common for both parents)
    min_x, min_y, max_x, max_y = get_bounds(parent0.problem.container.shape)
    x_length = abs(min_x - max_x)
    y_length = abs(min_y - max_y)

    # determine the minimum and maximum length in each dimension
    min_x_length = shape_min_length_proportion * x_length
    max_x_length = shape_max_length_proportion * x_length
    min_y_length = shape_min_length_proportion * y_length
    max_y_length = shape_max_length_proportion * y_length

    # determine the minimum area of the shape based on the container's area
    shape_min_area = shape_min_area_proportion * parent0.problem.container.shape.area

    partition_indices = [segment_index, _] = [0, 1]

    shape = None
    has_valid_shape = False
    i = 0

    # perform up to a certain number of attempts to create a partitioning shape that is (at least partially) within the container
    while i < max_attempt_num and not has_valid_shape:

        # randomly select the partitioning option
        partition_index = random.choice(partition_indices)
    
        # create a triangle/quadrilateral with a container-crossing segment
        if partition_index == segment_index:

            if bool(random.getrandbits(1)):
                shape = create_random_triangle_in_rectangle_corner(min_x, min_y, max_x, max_y)
            else:
                shape = create_random_quadrilateral_in_rectangle_corners(min_x, min_y, max_x, max_y)
        
        # create another shape
        else:
            
            # randomly determine the actual length in each dimension to use for a shape
            shape_x_length = random.uniform(min_x_length, max_x_length)
            shape_y_length = random.uniform(min_y_length, max_y_length)
            
            shape_indices = [circle_index, ellipse_index, _] = [0, 1, 2]
            
            # select a type of shape
            shape_index = random.choice(shape_indices)
            
            # create a circle
            if shape_index == circle_index:
                center = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                radius = random.choice([shape_x_length, shape_y_length]) * 0.5
                shape = Circle(center, radius)
                
            # create an ellipsis
            elif shape_index == ellipse_index:
                center = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                shape = Ellipse(center, shape_x_length, shape_y_length)
              
            # create a polygon  
            else:
                # randomly select the number of vertices for the polygon (at least 3 are needed, to have a triangle)
                vertex_num = random.randint(3, polygon_max_vertex_num)
                
                # randomly select a center for the bounding rectangle of the polygon
                polygon_bounds_center = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
                
                # determine the bounds of the polygon
                polygon_min_x = polygon_bounds_center[0] - shape_x_length * 0.5
                polygon_max_x = polygon_min_x + shape_x_length
                polygon_min_y = polygon_bounds_center[1] - shape_y_length * 0.5
                polygon_max_y = polygon_min_y + shape_y_length
                
                # create the polygon
                shape = create_random_polygon(polygon_min_x, polygon_min_y, polygon_max_x, polygon_max_y, vertex_num)
        
        # determine if the current shape is valid (it should have a minimum area and intersect with the container)
        has_valid_shape = shape is not None and shape.area >= shape_min_area and do_shapes_intersect(parent0.problem.container.shape, shape)
        
        i += 1

    # initially, each offspring is generated as the copy of one parent, before recombination
    offspring0 = copy.deepcopy(parent0)
    offspring1 = copy.deepcopy(parent1)

    # recombination is possible if there is a valid partitioning shape
    if has_valid_shape:

        # use the created shape to divide the placed items of each parent in 3 lists: items in the first region (inside the shape), items in the second region (outside the shape), and intersecting items
        parent0_separated_item_indices = list()
        parent1_separated_item_indices = list()
        for parent in [parent0, parent1]:
            region0_indices, region1_indices, intersected_indices = list(), list(), list()
            for item_index, placed_item in parent.placed_items.items():
                has_intersection = do_shapes_intersect(shape, placed_item.shape)
                if has_intersection:
                    intersected_indices.append(item_index)
                else:
                    if does_shape_contain_other(shape, placed_item.shape):
                        region0_indices.append(item_index)
                    else:
                        region1_indices.append(item_index)
            if parent == parent0:
                parent0_separated_item_indices = [region0_indices, region1_indices, intersected_indices]
            else:
                parent1_separated_item_indices = [region0_indices, region1_indices, intersected_indices]

        # for the first offspring, keep only the items placed in the first region in the first parent, then try to place the items of the second region of the second parent (duplication is internally prevented)
        for item_index in parent0_separated_item_indices[1] + parent0_separated_item_indices[2]:
            offspring0.remove_item(item_index)
        for item_index in parent1_separated_item_indices[1]:
            offspring0.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))

        # for the second offspring, keep only the items placed in the second region in the first parent, then try to place the items of the first region of the second parent (duplication is internally prevented)
        for item_index in parent0_separated_item_indices[0] + parent0_separated_item_indices[2]:
            offspring1.remove_item(item_index)
        for item_index in parent1_separated_item_indices[0]:
            offspring1.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))

        # find all the pairs (item index, parent index); one for each intersected item in a parent
        item_parent_pairs = list()
        for parent_index, intersected_item_indices in enumerate([parent0_separated_item_indices[2], parent1_separated_item_indices[2]]):
            for item_index in intersected_item_indices:
                item_parent_pairs.append((item_index, parent_index))

        # further actions are only needed if there are intersected items
        if item_parent_pairs:

            # generate the permutations of all (or a reasonable high number of) the possible orders of placing attempts for the items that were cut (do not handle duplications, since the addition function will discard them outright)
            permutations = list(itertools.islice(itertools.permutations(item_parent_pairs), max_permutation_num * 10))

            # if there are more permutations that the maximum possible, get a subset randomly
            if len(permutations) > max_permutation_num:
                permutations = random.choices(permutations, k=max_permutation_num)

            # each permutation produces a different offspring (based on one of the two base ones, chosen randomly) that tries to add items in the permutation's order; find the best one (for each base offspring) and all the ones that have a remarkable fitness value
            candidates = list()
            high_fitness_candidate_indices = list()
            best_candidate0_index = -1
            best_candidate1_index = -1
            for permutation_index, permutation in enumerate(permutations):
                derive_offspring_0 = bool(random.getrandbits(1))
                candidate_offspring = copy.deepcopy(offspring0) if derive_offspring_0 else copy.deepcopy(offspring1)
                for (item_index, parent_index) in permutation:
                    parent = parent0 if parent_index == 0 else parent1
                    candidate_offspring.add_item(item_index, parent.placed_items[item_index].position, parent.placed_items[item_index].rotation)
                candidate_fitness = get_fitness(candidate_offspring)
                best_candidate_index = best_candidate0_index if derive_offspring_0 else best_candidate1_index
                if best_candidate_index == -1 or candidate_fitness > get_fitness(candidates[best_candidate_index]):
                    if derive_offspring_0:
                        best_candidate0_index = permutation_index
                    else:
                        best_candidate1_index = permutation_index
                if candidate_fitness >= min_fitness_for_non_best:
                    high_fitness_candidate_indices.append(permutation_index)
                candidates.append(candidate_offspring)

            # locate the best candidates and make them selected; if no permutation produced an alternative version of the base offspring (one of them or both), use the old ones as final
            if best_candidate0_index != -1:
                offspring0 = candidates[best_candidate0_index]
            if best_candidate1_index != -1:
                offspring1 = candidates[best_candidate1_index]

            # if there are non-best alternative solutions with high fitness, return them together with the best
            if high_fitness_candidate_indices:
                return [offspring0, offspring1] + [candidates[index] for index in high_fitness_candidate_indices if index != best_candidate0_index and index != best_candidate1_index]

    # as a default case, return only the best pair of generated offspring
    return offspring0, offspring1


def mutate_with_addition(solution, max_attempt_num):

    """Try to mutate the passed solution in-place by adding an item"""

    # find the weight than can still be added to the container
    remaining_weight = solution.problem.container.max_weight - solution.weight

    # determine the feasible items: those that are within the capacity limits and not placed yet
    feasible_item_indices = [index for index in solution.problem.items.keys() if solution.problem.items[index].weight <= remaining_weight and index not in solution.placed_items.keys()]

    # only proceed if there are feasible items
    if feasible_item_indices:

        # determine the bounds of the container
        min_x, min_y, max_x, max_y = get_bounds(solution.problem.container.shape)

        # randomly select an item
        item_index = random.choice(feasible_item_indices)

        # perform up to a maximum number of attempts
        for _ in range(max_attempt_num):

            # if the action succeeds, there is nothing more to try
            if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360)):
                return True

    return False


def mutate_with_removal(solution):

    """Try to mutate the passed solution in-place by removing an item"""

    # removal is only possible if there are items placed in the container
    if solution.placed_items:

        # select an item randomly
        item_index = solution.get_random_placed_item_index()

        # remove the item
        return solution.remove_item(item_index)

    return False


def mutate_with_placement_modification(solution, max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, move_until_intersection_point_num, move_until_intersection_min_dist_proportion, rotate_until_intersection_angle_num, item_index=-1, action_index=-1):

    """Try to mutate the passed solution in-place by modifying an existing placement"""

    # placements must exist to be modified
    if solution.placed_items:

        # determine the bounds and side length of the container
        min_x, min_y, max_x, max_y = get_bounds(solution.problem.container.shape)
        x_length = abs(min_x - max_x)
        y_length = abs(min_y - max_y)

        # determine the maximum displacement that can be applied in small position changes, based on the container's side lengths
        max_small_x_change = small_position_change_proportion * x_length
        max_small_y_change = small_position_change_proportion * y_length

        # determine the maximum rotation that can be applied in small rotation changes
        max_small_rotation = 360 * small_rotation_change_proportion

        # determine the minimum and maximum relevant distance for move-until-intersection action
        move_until_intersection_min_dist = np.linalg.norm((x_length * move_until_intersection_min_dist_proportion, y_length * move_until_intersection_min_dist_proportion))
        move_until_intersection_max_dist = x_length + y_length

        # select a placed item randomly, if not imposed externally
        if item_index < 0:
            item_index = solution.get_random_placed_item_index()

        # swaps of properties between items will only be possible if there are at least two placements
        can_swap = len(solution.placed_items) >= 2

        # enumerate the possible actions
        position_change_index, small_position_change_index, move_until_intersect_index, position_swap_index, rotation_change_index, small_rotation_change_index, rotate_until_intersect_index, rotation_swap_index, position_rotation_change_index, small_position_rotation_change_index, position_rotation_swap_index = range(11)

        # if action index has not been externally imposed, calculate it
        if action_index < 0:

            # an item can always opt to have its position changed
            action_indices = [position_change_index, small_position_change_index, move_until_intersect_index]
            if can_swap:
                action_indices.append(position_swap_index)

            # only some items can be rotated
            if not np.isnan(solution.placed_items[item_index].rotation):
                action_indices.extend([rotation_change_index, small_rotation_change_index, rotate_until_intersect_index, position_rotation_change_index, small_position_rotation_change_index])
                if can_swap:
                    action_indices.extend([rotation_swap_index, position_rotation_swap_index])

            # select a modification action, with the same probability for all of the available ones
            action_index = random.choice(action_indices)

        # store the indices of items for which swapping should not be attempted, to be expanded on failure
        swap_ignore_indices = [item_index]

        # perform up to a maximum number of attempts
        for _ in range(max_attempt_num):

            has_modified = True

            # try to apply the action
            if action_index == position_change_index:
                has_modified = solution.move_item_to(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)))
            elif action_index == small_position_change_index:
                has_modified = solution.move_item(item_index, (random.uniform(-max_small_x_change, max_small_x_change), random.uniform(-max_small_y_change, max_small_y_change)))
            elif action_index == move_until_intersect_index:
                has_modified = solution.move_item_in_direction(item_index, (random.uniform(-1, 1), random.uniform(-1, 1)), move_until_intersection_point_num, move_until_intersection_min_dist, move_until_intersection_max_dist)
            elif action_index == position_swap_index:
                swap_item_index = solution.get_random_placed_item_index(swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(item_index, swap_item_index, swap_position=True, swap_rotation=False)
                swap_ignore_indices.append(swap_item_index)
            elif action_index == rotation_change_index:
                has_modified = solution.rotate_item_to(item_index, random.uniform(0, 360))
            elif action_index == small_rotation_change_index:
                has_modified = solution.rotate_item(item_index, random.uniform(-max_small_rotation, max_small_rotation))
            elif action_index == rotate_until_intersect_index:
                clockwise = bool(random.getrandbits(1))
                has_modified = solution.rotate_item_in_direction(item_index, clockwise, rotate_until_intersection_angle_num)
                if not has_modified:
                    has_modified = solution.rotate_item_in_direction(item_index, not clockwise, rotate_until_intersection_angle_num)
            elif action_index == rotation_swap_index:
                swap_item_index = solution.get_random_placed_item_index(swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(item_index, swap_item_index, swap_position=False, swap_rotation=True)
                swap_ignore_indices.append(swap_item_index)
            elif action_index == position_rotation_change_index:
                has_modified = solution.move_and_rotate_item_to(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))
            elif action_index == small_position_rotation_change_index:
                has_modified = solution.move_and_rotate_item(item_index, (random.uniform(-max_small_x_change, max_small_x_change), random.uniform(-max_small_y_change, max_small_y_change)), random.uniform(-max_small_rotation, max_small_rotation))
            elif action_index == position_rotation_swap_index:
                swap_item_index = solution.get_random_placed_item_index(swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(item_index, swap_item_index, swap_position=True, swap_rotation=True)
                swap_ignore_indices.append(swap_item_index)

            # if the action was successfully confirmed, there is nothing more to try
            if has_modified:
                return True

            # if the two possible directions for rotation-until-intersection have been (unsuccessfully) tried for the item, there is nothing more to do
            if action_index == rotate_until_intersect_index:
                return False

    return False


def get_mutation(solution, min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion,  mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob):

    """Generate and return a mutated copy of the passed solution"""

    # all changes will be applied to a copy of the individual
    mutated_solution = copy.deepcopy(solution)

    # temporarily keep the original solution as the best one
    best_solution = solution

    removal_num_to_compensate = 0
    iter_count = 0

    action_indices = [add_index, remove_index, _] = [0, 1, 2]
    action_weights = [mutation_add_weight, mutation_remove_weight, mutation_modify_weight]

    # perform iterations at least a minimum number of times and continue if removals need to be compensated
    while iter_count < min_iter_num or removal_num_to_compensate > 0:

        # after the minimum number of iterations, add items to compensate removals
        if iter_count >= min_iter_num:
            action_index = add_index

        # otherwise, perform a weighted choice of the next action to do
        else:
            action_index = random.choices(action_indices, action_weights, k=1)[0]

        # mutate with the selected action type
        if action_index == add_index:
            has_mutated = mutate_with_addition(mutated_solution, mutation_add_max_attempt_num)
            # if the addition is successful, it can compensate a previous removal; also count as compensated any failed additions after the base iteration limit, to avoid an eventual infinite loop
            if has_mutated or iter_count >= min_iter_num:
                removal_num_to_compensate = max(removal_num_to_compensate - 1, 0)
        elif action_index == remove_index:
            has_mutated = mutate_with_removal(mutated_solution)
            # if the removal is successful, it will need to be compensated
            if has_mutated:
                removal_num_to_compensate += 1
        else:
            has_mutated = mutate_with_placement_modification(mutated_solution, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion,  mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num)

        # if a mutation was applied and there is any possibility to select intermediate solutions as final ones and the current intermediate solution is better than any previous one, keep a copy of it
        if has_mutated and mutation_intermediate_selection_prob > 0 and get_fitness(mutated_solution) > get_fitness(best_solution):
            best_solution = copy.deepcopy(mutated_solution)

        iter_count += 1

    # if possible, select an intermediate solution as final with a certain probability if it is better than the last mutation solution
    if mutation_intermediate_selection_prob > 0 and mutated_solution != best_solution and best_solution != solution and random.uniform(0, 1) < mutation_intermediate_selection_prob:
        return best_solution
    # otherwise, keep the last mutated solution as final
    else:
        return mutated_solution


def generate_offspring(parents, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob, crossover_ignore_mutation_probability, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion, crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best, calculate_times=False):

    """Generate and return offspring from the passed parents"""

    start_time = crossover_time = mutation_time = 0

    # parents need to exist to generate offspring
    if not parents:

        return list()

    # if parents are represented as pairs, crossover should be applied before mutation
    can_use_crossover = type(parents[0]) == tuple

    # if there is no crossover, the parents will be the base for mutations
    individuals_to_mutate = parents if not can_use_crossover else list()

    # if possible, perform crossover
    if can_use_crossover:

        if calculate_times:
            start_time = time.time()

        # use crossover to generate (at least) two individuals per pair, that will be mutated later
        for (parent0, parent1) in parents:
            individuals_to_mutate.extend(get_crossover(parent0, parent1, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion, crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best))

        if calculate_times:
            crossover_time += get_time_since(start_time)

    offspring = list()

    if calculate_times:
        start_time = time.time()

    # mutate the individuals to get the final offspring
    for individual in individuals_to_mutate:

        # if the individual has been generated with crossover, ignore mutation altogether with a certain probability
        if can_use_crossover and random.uniform(0, 1) < crossover_ignore_mutation_probability:
            offspring.append(individual)

        else:
            # create a mutated copy of the individual
            mutated_individual = get_mutation(individual, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion,  mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob)

            # if crossover was used and the mutated individual is less fit (or same but loses the tie-break), keep the pre-mutation individual with a certain probability
            if can_use_crossover and get_fittest_solution([individual, mutated_individual]) == individual:
                offspring.append(individual)

            # in normal conditions, keep the mutated individual
            else:
                offspring.append(mutated_individual)

    if calculate_times:
        mutation_time += get_time_since(start_time)

    if calculate_times:
        return offspring, crossover_time, mutation_time

    return offspring


def get_surviving_population(population, survivor_num, population_update_pool_size):

    """Use tournament selection (with pools of the passed size) to restore a population with the passed number of survivors"""

    # if the number of survivors is equal to or higher than the current population size, return the whole population
    if survivor_num >= len(population):
        return population

    surviving_population = list()

    # otherwise, select as many survivors as needed
    for _ in range(survivor_num):

        # a tournament winner can survive
        survivor = get_tournament_winner(population, population_update_pool_size)

        # add the survivor to the new population
        surviving_population.append(survivor)

        # remove the survivor from the old population, to avoid duplicity
        population.remove(survivor)

    return surviving_population


def sort_by_fitness(population):

    """Sort the passed population by descending fitness, in-place, without considering tie-breaking criteria"""

    population.sort(key=lambda solution: get_fitness(solution), reverse=True)


def get_fitness_stats(population):

    """Return a dictionary with statistical information of the fitness of the passed population"""

    max_fitness = -np.inf
    min_fitness = np.inf
    fitness_sum = 0
    fitness_counts = dict()

    for solution in population:
        fitness = get_fitness(solution)
        fitness_sum += fitness
        if fitness > max_fitness:
            max_fitness = fitness
        if fitness < min_fitness:
            min_fitness = fitness
        if fitness in fitness_counts:
            fitness_counts[fitness] += 1
        else:
            fitness_counts[fitness] = 0

    return {"max": max_fitness, "min": min_fitness, "avg": fitness_sum / len(population), "mode": max(fitness_counts, key=fitness_counts.get)}


def solve_problem(problem, population_size=POPULATION_SIZE, initial_generation_item_specialization_iter_proportion=INITIAL_SOLUTION_GENERATION_FIRST_ITEM_SPECIALIZATION_ITER_PROPORTION, offspring_size=OFFSPRING_SIZE, elite_size=ELITE_SIZE, parent_selection_pool_size=PARENT_SELECTION_POOL_SIZE, population_update_pool_size=POPULATION_UPDATE_POOL_SIZE, max_generation_num=MAX_GENERATION_NUM, converge_generation_num=CONVERGE_GENERATION_NUM, mutation_min_iter_num=MUTATION_MIN_ITER_NUM, mutation_add_weight=MUTATION_ADD_WEIGHT, mutation_remove_weight=MUTATION_REMOVE_WEIGHT, mutation_modify_weight=MUTATION_MODIFY_WEIGHT, mutation_add_max_attempt_num=MUTATION_ADD_MAX_ATTEMPT_NUM, mutation_modify_max_attempt_num=MUTATION_MODIFY_MAX_ATTEMPT_NUM, small_position_change_proportion=MUTATION_MODIFY_SMALL_POSITION_CHANGE_PROPORTION, small_rotation_change_proportion=MUTATION_MODIFY_SMALL_ROTATION_CHANGE_PROPORTION, mutation_modify_move_until_intersection_point_num=MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM, mutation_modify_move_until_intersection_min_dist_proportion=MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION, mutation_modify_rotate_until_intersection_angle_num=MUTATION_MODIFY_ROTATE_UNTIL_INTERSECTION_ANGLE_NUM, mutation_intermediate_selection_prob=MUTATION_INTERMEDIATE_SELECTION_PROB, can_use_crossover=CAN_USE_CROSSOVER, crossover_ignore_mutation_probability=CROSSOVER_IGNORE_MUTATION_PROBABILITY, crossover_max_attempt_num=CROSSOVER_MAX_ATTEMPT_NUM, crossover_shape_min_length_proportion=CROSSOVER_SHAPE_MIN_LENGTH_PROPORTION, crossover_shape_max_length_proportion=CROSSOVER_SHAPE_MAX_LENGTH_PROPORTION, crossover_shape_min_area_proportion=CROSSOVER_SHAPE_MIN_AREA_PROPORTION, crossover_polygon_max_vertex_num=CROSSOVER_POLYGON_MAX_VERTEX_NUM, crossover_max_permutation_num=CROSSOVER_MAX_PERMUTATION_NUM, crossover_min_fitness_for_non_best_proportion=CROSSOVER_MIN_FITNESS_FOR_NON_BEST_PROPORTION, calculate_times=False, return_population_fitness_per_generation=False):

    """Find and return a solution to the passed problem, using an evolutionary algorithm"""

    start_time = initial_population_time = parent_selection_time = min_fitness_for_crossover_non_best_time = crossover_time = mutation_time = elite_finding_time = population_update_time = early_stop_time = fitness_per_generation_time = 0

    # a value representing a minimal difference of container's value for the problem, found scaling down the lowest value among items
    minimal_value_difference = min([item.value for item in problem.items.values()]) * 0.00001

    if calculate_times:
        start_time = time.time()

    # generate the initial population
    population = generate_population(problem, population_size, initial_generation_item_specialization_iter_proportion)

    if calculate_times:
        initial_population_time += get_time_since(start_time)

    if return_population_fitness_per_generation:
        population_fitness_per_generation = list()
    else:
        population_fitness_per_generation = None

    max_fitness = -np.inf
    iter_count_without_improvement = 0
    extended_population = None

    elite = list()

    # update the population up to a maximum number of generations
    for i in range(max_generation_num):

        # if needed, store the fitness of every individual of the population of the current generation
        if return_population_fitness_per_generation:

            if calculate_times:
                start_time = time.time()

            population_fitness_per_generation.append([get_fitness(solution) for solution in population])

            if calculate_times:
                fitness_per_generation_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # select as parents the individuals that will generate offspring
        parents = select_parents(population, offspring_size, parent_selection_pool_size, can_use_crossover)

        if calculate_times:
            parent_selection_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # calculate the minimum fitness for crossover non-best (alternative) solutions to be accepted as additional offspring; it is straightforward to find the threshold the population is sorted in descending fitness order, and an improvement in fitness (even if bare minimum) from a reference individual (according to the proportion) needs to happen
        crossover_min_fitness_for_non_best = 0
        if can_use_crossover and crossover_min_fitness_for_non_best_proportion > 0:
            need_to_improve_best = crossover_min_fitness_for_non_best_proportion >= 1
            sort_by_fitness(population)
            min_fitness_index = 0 if need_to_improve_best else math.ceil((1. - crossover_min_fitness_for_non_best_proportion) * len(population))
            crossover_min_fitness_for_non_best = get_fitness(population[min_fitness_index])
            crossover_min_fitness_for_non_best += minimal_value_difference

        if calculate_times:
            min_fitness_for_crossover_non_best_time += get_time_since(start_time)

        # generate offspring
        offspring_result = generate_offspring(parents, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob, crossover_ignore_mutation_probability, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion, crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best, calculate_times)
        if calculate_times:
            offspring, new_crossover_time, new_mutation_time = offspring_result
            crossover_time += new_crossover_time
            mutation_time += new_mutation_time
        else:
            offspring = offspring_result

        if calculate_times:
            start_time = time.time()

        # define a temporary extended population by joining the original population and their offspring
        extended_population = population + offspring

        # find the elite individuals (those with the highest fitness) among the extended population, sorted by descending fitness
        elite = get_fittest_solutions(extended_population, elite_size)

        if calculate_times:
            elite_finding_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # check if fitness has improved in this generation, to discard or contribute to confirm the assumption of convergence
        elite_fitness = get_fitness(elite[0]) if elite else -np.inf
        if elite_fitness > max_fitness:
            max_fitness = elite_fitness
            iter_count_without_improvement = 0
        else:
            iter_count_without_improvement += 1

        # stop early if the elite fitness has not improved for a certain number of iterations
        if iter_count_without_improvement >= converge_generation_num:
            break

        if calculate_times:
            early_stop_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # update the population, reserving places for the elite and selecting the rest from the extended population, to restore the standard population size
        population = elite + get_surviving_population([individual for individual in extended_population if individual not in elite], population_size - len(elite), population_update_pool_size)

        if calculate_times:
            population_update_time += get_time_since(start_time)

    # the fittest solution is the best one
    best_solution = elite[0] if elite else get_fittest_solution(population)

    # if needed, store the fitness of every individual of the last population (if stopped early, before selecting survivors, select them now)
    if return_population_fitness_per_generation:

        if calculate_times:
            start_time = time.time()

        if iter_count_without_improvement >= converge_generation_num:
            population = elite + get_surviving_population([individual for individual in extended_population if individual not in elite], population_size - len(elite), population_update_pool_size)

        population_fitness_per_generation.append([get_fitness(solution) for solution in population])

        if calculate_times:
            fitness_per_generation_time += get_time_since(start_time)

    # encapsulate all times informatively in a dictionary
    if calculate_times:
        approx_total_time = initial_population_time + parent_selection_time + min_fitness_for_crossover_non_best_time + crossover_time + mutation_time + elite_finding_time + population_update_time + early_stop_time + fitness_per_generation_time
        time_dict = {"Generation of the initial population": (initial_population_time, initial_population_time / approx_total_time), "Parent selection": (parent_selection_time, parent_selection_time / approx_total_time), "Min-fitness-for-crossover-non-best calculation": (min_fitness_for_crossover_non_best_time, min_fitness_for_crossover_non_best_time / approx_total_time), "Crossover": (crossover_time, crossover_time / approx_total_time), "Mutation": (mutation_time, mutation_time / approx_total_time), "Elite finding": (elite_finding_time, elite_finding_time / approx_total_time), "Population update": (population_update_time, population_update_time / approx_total_time), "Early stopping check": (early_stop_time, early_stop_time / approx_total_time), "Population fitness per generation gathering": (fitness_per_generation_time, fitness_per_generation_time / approx_total_time)}
        if return_population_fitness_per_generation:
            return best_solution, time_dict, population_fitness_per_generation
        return best_solution, time_dict

    if return_population_fitness_per_generation:
        return best_solution, population_fitness_per_generation

    return best_solution
