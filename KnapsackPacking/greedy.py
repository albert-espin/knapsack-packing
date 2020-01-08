import copy
import itertools
import random
import time
from problem_solution import Solution, get_bounds
from common_algorithm_functions import get_index_after_weight_limit, get_time_since

# default weight of the value of an item in the weighted sum of the value and the profitability ratio, that has a weight of 1-VALUE_WEIGHT
VALUE_WEIGHT = 0.5

# default weight of the area in the weighted product of item's weight and area; the weight's weight is 1-AREA_WEIGHT
AREA_WEIGHT = 0.5

# default maximum number of iterations for the greedy algorithm
MAX_ITER_NUM = 100000

# default maximum number of iterations without any change to perform early stopping
MAX_ITER_NUM_WITHOUT_CHANGES = 30000

# default number of repetitions of the algorithm
REPETITION_NUM = 1

# whether the constant score can be used if explicitely indicated
CAN_USE_CONSTANT_SCORE = True


def get_constant_score(item, value_weight, area_weight, can_use_constant_score=CAN_USE_CONSTANT_SCORE):

    """Return a constant score regardless of the item's characteristics or any weighting, if allowed"""

    if can_use_constant_score:
        return 1

    return get_weighted_sum_of_item_value_and_profitability_ratio(item, value_weight, area_weight)


def get_item_profitability_ratio(item, area_weight):

    """Return the profitability ratio of an item"""

    # the ratio is the value of the item divided by a weighted product of the weight and the shape's area
    return item.value / ((1. - area_weight) * item.weight + area_weight * item.shape.area)


def get_weighted_sum_of_item_value_and_profitability_ratio(item, value_weight, area_weight):

    """Return weighted sum of the value and the profitability ratio of an item"""

    return value_weight * item.value + (1. - value_weight) * get_item_profitability_ratio(item, area_weight)


def select_item(items_with_profit_ratio):

    """Given a list of tuples of the form (item_index, item_profitability_ratio, item), select an item proportionally to its profitability ratio, and return its index"""

    # find the cumulative profitability ratios, code based on random.choices() from the standard library of Python
    cumulative_profit_ratios = list(itertools.accumulate(item_with_profit_ratio[1] for item_with_profit_ratio in items_with_profit_ratio))
    profit_ratio_sum = cumulative_profit_ratios[-1]

    # randomly select a ratio within the range of the sum
    profit_ratio = random.uniform(0, profit_ratio_sum)

    # find the value that lies within the random ratio selected; binary search code is based on bisect.bisect_left from standard Python library, but adapted to profitability ratio check
    lowest = 0
    highest = len(items_with_profit_ratio)
    while lowest < highest:
        middle = (lowest + highest) // 2
        if cumulative_profit_ratios[middle] <= profit_ratio:
            lowest = middle + 1
        else:
            highest = middle

    return lowest


def solve_problem(problem, greedy_score_function=get_weighted_sum_of_item_value_and_profitability_ratio, value_weight=VALUE_WEIGHT, area_weight=AREA_WEIGHT, max_iter_num=MAX_ITER_NUM, max_iter_num_without_changes=MAX_ITER_NUM_WITHOUT_CHANGES, repetition_num=REPETITION_NUM, item_index_to_place_first=-1, item_specialization_iter_proportion=0., calculate_times=False, return_value_evolution=False):

    """Find and return a solution to the passed problem, using a greedy strategy"""

    # determine the bounds of the container
    min_x, min_y, max_x, max_y = get_bounds(problem.container.shape)

    max_item_specialization_iter_num = item_specialization_iter_proportion * max_iter_num

    start_time = 0
    sort_time = 0
    item_discarding_time = 0
    item_selection_time = 0
    addition_time = 0
    value_evolution_time = 0

    if calculate_times:
        start_time = time.time()

    if return_value_evolution:
        value_evolution = list()
    else:
        value_evolution = None

    if calculate_times:
        value_evolution_time += get_time_since(start_time)

    if calculate_times:
        start_time = time.time()

    # sort items (with greedy score calculated) by weight, to speed up their discarding (when they would cause the capacity to be exceeded)
    original_items_by_weight = [(index_item_tuple[0], greedy_score_function(index_item_tuple[1], value_weight, area_weight), index_item_tuple[1]) for index_item_tuple in sorted(list(problem.items.items()), key=lambda index_item_tuple: index_item_tuple[1].weight)]

    if calculate_times:
        sort_time += get_time_since(start_time)

    if calculate_times:
        start_time = time.time()

    # discard the items that would make the capacity of the container to be exceeded
    original_items_by_weight = original_items_by_weight[:get_index_after_weight_limit(original_items_by_weight, problem.container.max_weight)]

    if calculate_times:
        item_discarding_time += get_time_since(start_time)

    best_solution = None

    # if the algorithm is iterated, it is repeated and the best solution is kept in the end
    for _ in range(repetition_num):

        # if the algorithm is iterated, use a copy of the initial sorted items, to start fresh next time
        if repetition_num > 1:
            items_by_weight = copy.deepcopy(original_items_by_weight)
        else:
            items_by_weight = original_items_by_weight

        # create an initial solution with no item placed in the container
        solution = Solution(problem)

        # placements can only be possible with capacity and valid items
        if problem.container.max_weight and items_by_weight:

            iter_count_without_changes = 0

            # try to add items to the container, for a maximum number of iterations
            for i in range(max_iter_num):

                if calculate_times:
                    start_time = time.time()

                # if needed, select a specific item to try to place (only for a maximum number of attempts)
                if item_index_to_place_first >= 0 and i < max_item_specialization_iter_num:
                    item_index = item_index_to_place_first
                    list_index = -1

                # perform a random choice of the next item to try to place, weighting each item with their profitability ratio, that acts as an stochastic selection probability
                else:
                    list_index = select_item(items_by_weight)
                    item_index = items_by_weight[list_index][0]

                if calculate_times:
                    item_selection_time += get_time_since(start_time)

                if calculate_times:
                    start_time = time.time()

                # try to add the item in a random position and with a random rotation; if it is valid, remove the item from the pending list
                if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360)):

                    # the item to place first is assumed to have been placed, if there was any
                    item_index_to_place_first = -1

                    if calculate_times:
                        addition_time += get_time_since(start_time)

                    # find the weight that can still be added
                    remaining_weight = problem.container.max_weight - solution.weight

                    # stop early if the capacity has been exactly reached
                    if not remaining_weight:
                        break

                    # remove the placed item from the list of pending items
                    if list_index >= 0:
                        items_by_weight.pop(list_index)

                    # if focusing on an item to place first, find its associated entry in the list to remove it
                    else:
                        for list_i in range(len(items_by_weight)):
                            if items_by_weight[list_i][0] == item_index:
                                items_by_weight.pop(list_i)
                                break

                    if calculate_times:
                        start_time = time.time()

                    # discard the items that would make the capacity of the container to be exceeded
                    items_by_weight = items_by_weight[:get_index_after_weight_limit(items_by_weight, remaining_weight)]

                    if calculate_times:
                        item_discarding_time += get_time_since(start_time)

                    # stop early if it is not possible to place more items, because all have been placed or all the items outside would cause the capacity to be exceeded
                    if not items_by_weight:
                        break

                    # reset the potential convergence counter, since an item has been added
                    iter_count_without_changes = 0

                else:

                    if calculate_times:
                        addition_time += get_time_since(start_time)

                    # register the fact of being unable to place an item this iteration
                    iter_count_without_changes += 1

                    # stop early if there have been too many iterations without changes (unless a specific item is tried to be placed first)
                    if iter_count_without_changes >= max_iter_num_without_changes and item_index_to_place_first < 0:
                        break

                if return_value_evolution:

                    if calculate_times:
                        start_time = time.time()

                    value_evolution.append(solution.value)

                    if calculate_times:
                        value_evolution_time += get_time_since(start_time)

        # if the algorithm uses multiple iterations, adopt the current solution as the best one if it is the one with highest value up to now
        if not best_solution or solution.value > best_solution.value:
            best_solution = solution

    # encapsulate all times informatively in a dictionary
    if calculate_times:
        approx_total_time = sort_time + item_selection_time + item_discarding_time + addition_time + value_evolution_time
        time_dict = {"Weight-sort and profit ratio calculation": (sort_time, sort_time / approx_total_time), "Stochastic item selection": (item_selection_time, item_selection_time / approx_total_time), "Item discarding": (item_discarding_time, item_discarding_time / approx_total_time), "Addition and geometric validation": (addition_time, addition_time / approx_total_time), "Keeping value of each iteration": (value_evolution_time, value_evolution_time / approx_total_time)}
        if return_value_evolution:
            return best_solution, time_dict, value_evolution
        return best_solution, time_dict

    if return_value_evolution:
        return best_solution, value_evolution

    return best_solution
