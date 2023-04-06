from unittest import TestCase, main
from revolver.utils.domain import ClosedInterval, ClosedList, is_element
from revolver.utils.transformer import denormalize_index
import numpy as np


class TestIsElement(TestCase):
    def test_int(self):
        self.assertTrue(is_element(1, int))
        self.assertFalse(is_element(1.0, int))
        self.assertFalse(is_element('test_str', float))

    def test_float(self):
        self.assertTrue(is_element(1, float))
        self.assertTrue(is_element(1.0, float))
        self.assertFalse(is_element('test_str', float))

    def test_complex(self):
        self.assertTrue(is_element(1, complex))
        self.assertTrue(is_element(1.0, complex))
        self.assertTrue(is_element(1j, complex))
        self.assertFalse(is_element('test_str', float))

    def test_in_iterable(self):
        self.assertTrue(is_element(1, {1, 3, 5}))
        self.assertTrue(is_element(3, (1, 3, 5)))
        self.assertTrue(is_element(5, [1, 3, 5]))
        self.assertTrue(is_element(7.0, [1, 3, 7.0]))
        self.assertTrue(is_element('a', "test case"))

    def test_not_in_iterable(self):
        self.assertFalse(is_element((3, 5), (1, 3, 5)))
        self.assertFalse(is_element({1}, {1, 3, 5}))
        self.assertFalse(is_element(7, [1, 3, 5]))
        self.assertFalse(is_element(None, []))


class TestClosedList(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elements = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24)

    def test_constructor(self):
        with self.assertRaises(TypeError):
            ClosedList(None)
        with self.assertRaises(TypeError):
            ClosedList(3)

    def test_get_item(self):
        closed_list = ClosedList(self.elements)
        self.assertEqual(24, closed_list[1.0])
        self.assertEqual(8, closed_list[1])
        self.assertEqual(24, closed_list[-1])
        self.assertEqual(self.elements[3], closed_list[3])

    def test_index_error(self):
        closed_list = ClosedList(self.elements)
        with self.assertRaises(IndexError):
            closed_list[1.5]
        with self.assertRaises(IndexError):
            closed_list[len(self.elements) * 5]

    def test_pick_random(self):
        elements = self.elements
        closed_list = ClosedList(elements)
        np.random.seed(42)
        float_index, element = closed_list.pick_random()
        self.assertTrue(element in elements)
        index = denormalize_index(float_index, len(closed_list))
        self.assertEqual(element, elements[index])

    def test_pick_random_neighbor(self):
        elements = self.elements
        closed_list = ClosedList(elements)
        float_index, element = closed_list.pick_random_neighbor(0.75, 0.25)
        self.assertTrue(1.0 >= float_index >= 0.5)


class TestClosedInterval(TestCase):
    def test_constructor(self):
        with self.assertRaises(TypeError):
            ClosedInterval(None, 2, None)

    def test_get_item(self):
        interval = ClosedInterval(0.5, 2.5)
        self.assertEqual(2.5, interval[1.0])
        self.assertEqual(1.5, interval[1])
        self.assertEqual(2.5, interval[-1])

    def test_index_error(self):
        interval = ClosedInterval(0.5, 2.5)
        with self.assertRaises(IndexError):
            interval[1.5]
        with self.assertRaises(IndexError):
            interval[3]

    def test_float_type(self):
        interval = ClosedInterval(0.5, 24, float)
        self.assertTrue(is_element(1, interval))
        self.assertFalse(is_element(0, interval))
        self.assertFalse(is_element('test_str', interval))

    def test_float_range(self):
        interval = ClosedInterval(0.5, 0.7)
        passed = True
        for i in range(100):
            n, r = interval.pick_random()
            passed = passed and (0.5 <= r <= 0.7)
        self.assertTrue(passed)
        for i in range(100):
            n, r = interval.pick_random_neighbor(0.5, 0.25)
            passed = passed and (0.55 <= r <= 0.65)
        self.assertTrue(passed)

    def test_int_type(self):
        interval = ClosedInterval(0.5, 24, int)
        self.assertFalse(is_element(1.0, interval))
        self.assertFalse(is_element(0, interval))
        self.assertFalse(is_element('str', interval))
        self.assertTrue(is_element(3, interval))

    def test_int_range(self):
        interval = ClosedInterval(0, 10)
        passed = True
        for i in range(100):
            n, r = interval.pick_random()
            passed = passed and (0 <= r <= 10)
        self.assertTrue(passed)
        for i in range(100):
            n, r = interval.pick_random_neighbor(0.5, 0.2)
            passed = passed and (3 <= r <= 7)
        self.assertTrue(passed)


if __name__ == '__main__':
    main()
