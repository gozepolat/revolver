# -*- coding: utf-8 -*-
from unittest import main
from unittest import TestCase
from revolver.meta.scope import Scoped, get_elements


class ScopedChild(Scoped):
    def __init__(self, scope, m1, *args, **kwargs):
        super(ScopedChild, self).__init__(scope, *args, **kwargs)
        self.m1 = m1


class ScopedGrandchild(ScopedChild):
    def __init__(self, scope, m1, m2, m3, *args, **kwargs):
        super(ScopedGrandchild, self).__init__(scope, m1, *args, **kwargs)
        self.m2 = m2
        self.m3 = m3


class ScopedGreatGrandchild(ScopedGrandchild):
    def __init__(self, scope, m1, m2, m3, m4, *args, **kwargs):
        super(ScopedGreatGrandchild, self).__init__(scope, m1, m2, m3, *args, **kwargs)
        self.m4 = m4


class TestScopedChildDescendants(TestCase):
    def test_same_types_with_different_scopes(self):
        child1 = ScopedChild('scope1', 4)
        child2 = ScopedChild('scope2', 5)
        self.assertNotEqual(child1.m1, child2.m1)

        self.assertEqual(child1.scope, 'scope1')
        self.assertNotEqual(child1.scope, child2.scope)

        child1 = ScopedGrandchild('grand_scope1', 4, 6, 8)
        child2 = ScopedGrandchild('grand_scope2', 5, 7, 9)
        self.assertNotEqual(child1.m1, child2.m1)

    def test_has_random_key(self):
        ScopedChild('_random', 4)
        has_random_key = False
        for k in get_elements().keys():
            # check if scope is now something like _random021391238
            if '_random' == k[:7] and len(k) > 7:
                has_random_key = True
                break
        self.assertTrue(has_random_key)

    def test_same_types_with_the_same_scope(self):
        child1 = ScopedChild('scope', 4)
        child2 = ScopedChild('scope', 5)
        self.assertEqual(child1.m1, child2.m1)

        child1.m1 = 98
        self.assertEqual(child2.m1, 98)

    def test_same_grand_types_with_the_same_scope(self):
        child1 = ScopedGrandchild('grand_scope', 4, 6, 8)
        child2 = ScopedGrandchild('grand_scope', 5, 7, 9)
        self.assertEqual(child1.m1, child2.m1)
        self.assertTrue(child1.scope == child2.scope == 'grand_scope')

        child1.m3 = 987
        self.assertEqual(child2.m3, 987)

    def test_same_great_grand_types_with_the_same_scope(self):
        child1 = ScopedGreatGrandchild('great_grand_scope', 4, 6, 8, 10)
        child2 = ScopedGreatGrandchild('great_grand_scope', 5, 7, 9, 11)
        self.assertEqual(child1.m1, child2.m1)
        self.assertTrue(child1.scope == child2.scope == 'great_grand_scope')

        child1.m1 = 98
        self.assertEqual(child2.m1, 98)

        child1.m3 = 987
        self.assertEqual(child2.m3, 987)

        child1.m4 = 9876
        self.assertEqual(child2.m4, 9876)

    def test_different_types_with_the_same_scope(self):
        ScopedGreatGrandchild('5', 3, 4, 5, 6)
        with self.assertRaises(TypeError):
            ScopedGrandchild('5', 3, 4, 5)
        with self.assertRaises(TypeError):
            ScopedChild('5', 3)

    def test_invalid_scope(self):
        with self.assertRaises(TypeError):
            ScopedGrandchild(5, 3, 4, 5)
        with self.assertRaises(TypeError):
            ScopedChild(3)
        with self.assertRaises(ValueError):
            ScopedChild("", 3)


if __name__ == '__main__':
    main()
