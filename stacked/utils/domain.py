# -*- coding: utf-8 -*-
import common
from logging import warning
import numpy as np
from transformer import normalize_index, denormalize_index, normalize_float, denormalize_float


def log(log_func, msg):
    if common.DEBUG_DOMAIN:
        log_func("stacked.utils.domain: %s" % msg)


def is_inside(value, domain):
    """ Check whether the value belongs to a non-type domain """
    try:
        inside = value in domain
        return inside
    except TypeError:
        log(warning, "{} can not be an element of {}".format(value, domain))
        return False


def is_element(value, domain):
    """ Check if a value belongs to the given domain """
    if type(domain) == type:
        if domain == complex:  # subset
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, complex):
                return True
            log(warning, "{} type is not supported for complex".format(type(value)))
            return False
        elif domain == float and isinstance(value, int):
            return True
        return isinstance(value, domain)
    return is_inside(value, domain)


class Domain(object):
    """ Interface for the domain """

    def pick_random(self):
        """ Pick a random element from the domain (discrete uniform distribution)

        Return a pair (normalized float value, real value)
        normalized float value is always between 0 and 1 """
        raise NotImplementedError("Random_element method is not yet implemented!")

    def pick_random_neighbor(self, normalized_float, normalized_diameter, picker=None):
        """ Pick a random neighbor no further away than the diameter """
        raise NotImplementedError("Mutate method is not yet implemented!")


class ClosedList(Domain):
    """ A closed domain that consists of discrete elements

    pick* methods will not return elements outside the domain,
    they will return a pair of both the normalized index,
    and the stored element"""

    def __init__(self, elements, mapper=None):
        self.elements = elements
        self.mapper = mapper
        self.cardinality = len(elements)
        assert (self.cardinality > 0)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.elements[index]
        index = denormalize_index(index, self.__len__())
        return self.elements[index]

    def __contains__(self, item):
        return is_element(item, self.elements)

    def _pick_within_distance(self, normalized_float, diameter):
        """ Pick a random neighbor no further away than the diameter

          diameter normalized similar to the given value
         """
        assert (1.0 >= diameter >= 0)
        num = self.cardinality
        if diameter == 0:
            log(warning, "Diameter is zero, picking the only possible value")
            index = denormalize_index(normalized_float, num, self.mapper)
        else:

            high = denormalize_index(normalized_float + diameter, num, self.mapper)
            low = denormalize_index(max(0.0, normalized_float - diameter), num, self.mapper)
            if low == high:
                log(warning, "Diameter is too small {}, picking the only possible value".format(diameter))
                index = high
            else:
                index = np.random.randint(low, high + 1)
        if index < num:
            return normalize_index(index, num, self.mapper), self.elements[index]
        log(warning, "{} not in domain, index {} too large, picking random".format(normalized_float, index))
        return self.pick_random()

    def get_normalized_index(self, element, compare_fn=lambda x, y: x == y):
        index = -1
        for i, e in enumerate(self.elements):
            if compare_fn(e, element):
                index = i
                break
        if index < 0:
            return -1.0
        return normalize_index(index, self.cardinality, self.mapper)

    def pick_random(self):
        """ Pick a random element from the domain (discrete uniform distribution)

        :return a pair (normalized float, real value)"""
        index = np.random.randint(self.cardinality)
        normalized_float = normalize_index(index, self.cardinality, self.mapper)
        return normalized_float, self.elements[index]

    def pick_random_neighbor(self, normalized_float, normalized_diameter, picker=None):
        """ Pick a random neighbor no further away than the diameter

         normalized_float float version of the index
         normalized_diameter around the normalized_float, around which a random neighbor can be picked
         picker if not None, picker should return the new normalized_float and the neighbor value"""
        if self.cardinality == 1:
            return self.elements[0]
        if picker is None:
            return self._pick_within_distance(normalized_float, normalized_diameter)
        return picker(self.elements, normalized_float, normalized_diameter, self.mapper)


class ClosedInterval(Domain):
    """ Ordered closed interval of element_type int or float

    e.g. is_element(3, ClosedInterval(1.0, 6.0)) # True
    If element type is None, uses the type of the bounds
    """
    def __init__(self, lower_bound, upper_bound, element_type=None, mapper=None):
        super(ClosedInterval, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        assert(self.lower_bound <= self.upper_bound)

        self.element_type = element_type
        self.mapper = mapper

        if element_type is None:
            if not isinstance(lower_bound, type(upper_bound)):
                raise TypeError("Lower bound is not the same type as upper bound")
            self.element_type = type(lower_bound)
        assert(self.element_type == int or self.element_type == float)

    def __contains__(self, item):
        return is_element(item, self.element_type) and self.lower_bound <= item <= self.upper_bound

    def __len__(self):
        return self.upper_bound - self.lower_bound + 1

    def __getitem__(self, index):
        if isinstance(index, int):
            if index + self.lower_bound > self.upper_bound:
                raise IndexError("Interval integer index out of range")
            if index < 0:
                index = self.__len__() + index
            return index + self.lower_bound
        if not 1.0 >= index >= 0.0:
            raise IndexError("Interval float index out of range (not normalized properly)")
        return denormalize_float(index, self.lower_bound, self.upper_bound, self.mapper)

    def _pick_random_int_neighbor(self, center, diameter):
        assert (self.element_type == int)
        lower_index = denormalize_index(max(center - diameter, 0), self.__len__(), self.mapper)
        upper_index = denormalize_index(center + diameter, self.__len__(), self.mapper)
        upper_index = min(upper_index, self.__len__() - 1)
        if lower_index < upper_index:
            index = np.random.randint(lower_index, upper_index + 1)
        else:
            log(warning, "Diameter is too small {}, picking the only possible value".format(diameter))
            assert (lower_index == upper_index)
            index = lower_index
        value = index + self.lower_bound
        center = normalize_float(value, self.lower_bound, self.upper_bound, self.mapper)
        return center, value

    def get_normalized_index(self, element, *_):
        index = element - self.lower_bound
        if index >= self.__len__() or index < 0:
            return -1
        return normalize_index(index, self.__len__(), self.mapper)

    def pick_random(self):
        """ Pick a random element from the domain (uniform distribution) """
        if self.element_type == int:
            value = np.random.randint(self.lower_bound, self.upper_bound + 1)
            normalized_float = normalize_float(value, self.lower_bound, self.upper_bound, self.mapper)
            return normalized_float, value
        assert (self.element_type == float)
        normalized_float = np.random.random()
        value = denormalize_float(normalized_float, self.lower_bound, self.upper_bound, self.mapper)
        return normalized_float, value

    def pick_random_neighbor(self, center, diameter, picker=None):
        """ Pick a random neighbor around the normalized center no further away than the normalized diameter """
        if picker is not None:
            return picker(center, diameter, self.mapper)
        if self.element_type == float:
            center = np.random.uniform(center - diameter,
                                       center + diameter)
            return center, denormalize_float(center, self.lower_bound, self.upper_bound, self.mapper)
        return self._pick_random_int_neighbor(center, diameter)

