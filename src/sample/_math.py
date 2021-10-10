from fractions import Fraction
from random import Random
from typing import List, Tuple, TypeVar

from wai.common.math.sets import *

_T = TypeVar("_T")


def random_subset(
        collection: List[_T],
        subset_size: int,
        random: Random = Random(),
        order_matters: bool = False
) -> List[_T]:
    """
    Generates a random subset of the given collection, of the
    given size.

    :param collection:
                The collection to select from.
    :param subset_size:
                The number of items to select.
    :param random:
                An optional source of randomness.
    :param order_matters:
                Whether the order of selection matters.
    :return:
                The subset, as a list if order matters or
                a set if not.
    """
    size = len(collection)
    num_subsets = number_of_subsets(size, subset_size, order_matters)
    subset_number = random.randrange(num_subsets)
    indices = subset_number_to_subset(size, subset_size, subset_number, order_matters)
    return list(collection[index] for index in indices)


def random_permutation(collection: List[_T], random: Random = Random()) -> List[_T]:
    """
    Generates a random permutation of the given collection.

    :param collection:
                The collection to generate a permutation of.
    :param random:
                An optional source of randomness.
    :return:
                A random permutation of the collection.
    """
    return random_subset(collection, len(collection), random, True)


def calculate_schedule(ratios: Tuple[int]) -> List[int]:
    """
    Calculates the schedule of labels to return to best split
    items into the bins.

    :return:    The label schedule.
    """
    # Initialise an empty schedule
    schedule: List[int] = []

    # The initial best candidate binning is all bins empty
    best_candidate: Tuple[int] = tuple(0 for _ in range(len(ratios)))

    # The schedule cycle-length is the sum of ratios
    i = 0
    for schedule_index in range(sum(ratios)):
        print(i)
        i+=1
        # Create a candidate ratio for each of the possible binnings
        # (each being a single item added to one of the bins)
        candidate_ratios: Tuple[Tuple[int, ...]] = tuple(
            tuple(ratio + 1 if i == candidate_index else ratio
                  for i, ratio in enumerate(best_candidate))
            for candidate_index in range(len(ratios))
        )

        # Calculate the integer dot-product of each candidate ratio
        # to determine which is closest to the desired ratio
        candidate_dps: Tuple[Fraction, ...] = tuple(
            integer_dot_product(ratios, candidate_ratio)
            for candidate_ratio in candidate_ratios
        )

        # Select the candidate with the best (greatest) dot-product
        best_candidate_index = None
        best_candidate_dp = None
        for candidate_index, candidate_dp in enumerate(candidate_dps):
            if best_candidate_index is None or candidate_dp > best_candidate_dp:
                best_candidate = candidate_ratios[candidate_index]
                best_candidate_index = candidate_index
                best_candidate_dp = candidate_dp

        # Add the selected candidate bin to the schedule
        schedule.append(best_candidate_index)

    return schedule


def integer_dot_product(a: Tuple[int], b: Tuple[int]) -> Fraction:
    """
    Calculates the square of the dot-product between to vectors
    of integers.

    :param a:   The first integer vector.
    :param b:   The second integer vector.
    :return:    The square of the dot-product.
    """
    # Make sure the vectors are the same length
    if len(a) != len(b):
        raise ValueError(f"Can't perform integer dot product between vectors of different "
                         f"lengths: {a}, {b}")

    return Fraction(
        sum(a_i * b_i for a_i, b_i in zip(a, b)) ** 2,
        sum(a_i ** 2 for a_i in a) * sum(b_i ** 2 for b_i in b)
    )
