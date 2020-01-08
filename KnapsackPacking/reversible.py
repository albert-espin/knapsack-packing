import copy
import random
import time
from problem_solution import Solution, get_bounds
from common_algorithm_functions import get_index_after_weight_limit, get_time_since

# default maximum number of iterations for the reversible algorithm
MAX_ITER_NUM = 100000

# default maximum number of iterations without any addition to perform early stopping
MAX_ITER_NUM_WITHOUT_ADDITIONS = 30000

# default number of iterations to revert a removal operation if it has not increased the value in the container
ITER_NUM_TO_REVERT_REMOVAL = 200

# probability of removing an item in an iteration
ITEM_REMOVAL_PROBABILITY = 0.005

# probability of removing an item when another removal is still recent
CONSECUTIVE_ITEM_REMOVAL_PROBABILITY = 0.001

# probability to completely avoid trying to place again a removed item until its removal becomes reverted or permanently accepted
IGNORE_REMOVED_ITEM_PROBABILITY = 0.5

# probability of modifying an item placement (moving or rotating) in an iteration
PLACEMENT_MODIFICATION_PROBABILITY = 0.05


def select_item(items_by_weight):

    # perform a random choice of the next item to try to place
    list_index = random.randint(0, len(items_by_weight) - 1)
    item_index = items_by_weight[list_index][0]

    return list_index, item_index


def solve_problem(problem, max_iter_num=MAX_ITER_NUM, max_iter_num_without_adding=MAX_ITER_NUM_WITHOUT_ADDITIONS, iter_num_to_revert_removal=ITER_NUM_TO_REVERT_REMOVAL, remove_prob=ITEM_REMOVAL_PROBABILITY, consec_remove_prob=CONSECUTIVE_ITEM_REMOVAL_PROBABILITY, ignore_removed_item_prob=IGNORE_REMOVED_ITEM_PROBABILITY, modify_prob=PLACEMENT_MODIFICATION_PROBABILITY, calculate_times=False, return_value_evolution=False):

    """Find and return a solution to the passed problem, using an reversible strategy"""

    # create an initial solution with no item placed in the container
    solution = Solution(problem)

    # determine the bounds of the container
    min_x, min_y, max_x, max_y = get_bounds(problem.container.shape)

    start_time = 0
    sort_time = 0
    item_discarding_time = 0
    item_selection_time = 0
    addition_time = 0
    removal_time = 0
    modification_time = 0
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

    # sort items by weight, to speed up their discarding (when they would cause the capacity to be exceeded)
    items_by_weight = sorted(list(problem.items.items()), key=lambda index_item_tuple: index_item_tuple[1].weight)

    if calculate_times:
        sort_time += get_time_since(start_time)

    iter_count_since_addition = 0
    iter_count_since_removal = 0
    solution_before_removal = None

    if calculate_times:
        start_time = time.time()

    # discard the items that would make the capacity of the container to be exceeded
    items_by_weight = items_by_weight[:get_index_after_weight_limit(items_by_weight, problem.container.max_weight)]

    ignored_item_index = -1

    if calculate_times:
        item_discarding_time += get_time_since(start_time)

    # placements can only be possible with capacity and valid items
    if problem.container.max_weight and items_by_weight:

        # try to add items to the container, for a maximum number of iterations
        for i in range(max_iter_num):

            if calculate_times:
                start_time = time.time()

            # perform a random choice of the next item to try to place
            list_index, item_index = select_item(items_by_weight)

            if calculate_times:
                item_selection_time += get_time_since(start_time)

            if calculate_times:
                start_time = time.time()

            # try to add the item in a random position and with a random rotation; if it is valid, remove the item from the pending list
            if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360)):

                if calculate_times:
                    addition_time += get_time_since(start_time)

                # find the weight that can still be added
                remaining_weight = problem.container.max_weight - solution.weight

                # stop early if the capacity has been exactly reached
                if not remaining_weight:
                    break

                # remove the placed item from the list of pending items
                items_by_weight.pop(list_index)

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
                iter_count_since_addition = 0

            else:

                if calculate_times:
                    addition_time += get_time_since(start_time)

                # register the fact of being unable to place an item this iteration
                iter_count_since_addition += 1

                # stop early if there have been too many iterations without changes
                if iter_count_since_addition == max_iter_num_without_adding:
                    break

            if calculate_times:
                start_time = time.time()

            # if there are items in the container, try to remove an item with a certain probability (different if there was a recent removal)
            if solution.weight > 0 and random.uniform(0., 1.) < (consec_remove_prob if solution_before_removal else remove_prob):

                # if there is no solution prior to a removal with pending re-examination
                if not solution_before_removal:

                    # save the current solution before removing, just in case in needs to be restored later
                    solution_before_removal = copy.deepcopy(solution)

                    # reset the counter of iterations since removal, to avoid reverting earlier than needed
                    iter_count_since_removal = 0

                # get the index of the removed item, which is randomly chosen
                removed_index = solution.remove_random_item()

                # with a certain probability, only if not ignoring any item yet, ignore placing again the removed item until the operation gets reverted or permanently accepted
                if ignored_item_index < 0 and items_by_weight and random.uniform(0., 1.) < ignore_removed_item_prob:
                    ignored_item_index = removed_index

                # otherwise, add the removed item to the weight-sorted list of pending-to-add items
                else:
                    items_by_weight.insert(get_index_after_weight_limit(items_by_weight, problem.items[removed_index].weight), (removed_index, problem.items[removed_index]))

            # if there is a recent removal to be confirmed or discarded after some time
            if solution_before_removal:

                # re-examine a removal after a certain number of iterations
                if iter_count_since_removal == iter_num_to_revert_removal:

                    # if the value in the container has improved since removal, accept the operation in a definitive way
                    if solution.value > solution_before_removal.value:

                        # if an item had been ignored, make it available for placement again
                        if ignored_item_index >= 0:
                            items_by_weight.insert(get_index_after_weight_limit(items_by_weight, problem.items[ignored_item_index].weight), (ignored_item_index, problem.items[ignored_item_index]))

                    # otherwise, revert the solution to the pre-removal state
                    else:
                        solution = solution_before_removal

                        # after reverting a removal, have some margin to try to add items
                        iter_count_since_addition = 0

                    # reset removal data
                    solution_before_removal = None
                    iter_count_since_removal = 0
                    ignored_item_index = -1

                # the check will be done after more iterations
                else:
                    iter_count_since_removal += 1

            if calculate_times:
                removal_time += get_time_since(start_time)

            if calculate_times:
                start_time = time.time()

            # if there are still items in the container (maybe there was a removal), modify existing placements with a certain probability
            if solution.weight > 0 and random.uniform(0., 1.) < modify_prob:

                # perform a random choice of the item to try to affect
                _, item_index = select_item(items_by_weight)

                # move to a random position of the container with a probability of 50%
                if random.uniform(0., 1.) < 0.5:
                    solution.move_item_to(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)))

                # otherwise, perform a random rotation
                else:
                    solution.rotate_item_to(item_index, random.uniform(0, 360))

            if calculate_times:
                modification_time += get_time_since(start_time)

            if return_value_evolution:

                if calculate_times:
                    start_time = time.time()

                value_evolution.append(solution.value)

                if calculate_times:
                    value_evolution_time += get_time_since(start_time)

        # in the end, revert the last unconfirmed removal if it did not improve the container's value
        if solution_before_removal and solution.value < solution_before_removal.value:
            solution = solution_before_removal

            if return_value_evolution:

                if calculate_times:
                    start_time = time.time()

                value_evolution[-1] = solution.value

                if calculate_times:
                    value_evolution_time += get_time_since(start_time)

    # encapsulate all times informatively in a dictionary
    if calculate_times:
        approx_total_time = sort_time + item_selection_time + item_discarding_time + addition_time + removal_time + modification_time + value_evolution_time
        time_dict = {"Weight-sort": (sort_time, sort_time / approx_total_time), "Stochastic item selection": (item_selection_time, item_selection_time / approx_total_time), "Item discarding": (item_discarding_time, item_discarding_time / approx_total_time), "Addition (with geometric validation)": (addition_time, addition_time / approx_total_time), "Removal and reverting-removal": (removal_time, removal_time / approx_total_time), "Placement modification (with geometric validation)": (modification_time, modification_time / approx_total_time), "Keeping value of each iteration": (value_evolution_time, value_evolution_time / approx_total_time)}
        if return_value_evolution:
            return solution, time_dict, value_evolution
        return solution, time_dict

    if return_value_evolution:
        return solution, value_evolution

    return solution
