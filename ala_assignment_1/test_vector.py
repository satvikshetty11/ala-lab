"""
test_vector.py — Full test suite for vec.py

Covers every public method of Vec:
    Construction        : __init__
    Arithmetic          : __add__, __sub__, __neg__, __rmul__, __imul__,
                          __radd__, __iadd__
    Built-ins           : __len__, __repr__
    Factory methods     : zeros, ones, uniform
    Math operations     : norm, mean, demean, std

Lecture-derived properties (demean-std.tex / AME5101 Unit 1) are tested
explicitly for mean, demean, and std.

Run:
    python -m unittest test_vector -v          (from src/vec/)
    python -m unittest discover -s src/vec -v  (from project root)
"""

import math
import unittest
from vec import Vec


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

EPS = 1e-9


def approx(a: float, b: float, tol: float = EPS) -> bool:
    return abs(a - b) < tol


def vec_approx(u: Vec, v: Vec, tol: float = EPS) -> bool:
    if len(u) != len(v):
        return False
    return all(abs(a - b) < tol for a, b in zip(u.elements, v.elements))


# ===========================================================================
# 1. Construction  (__init__)
# ===========================================================================

class TestInit(unittest.TestCase):

    def test_from_list_of_ints(self):
        v = Vec([1, 2, 3])
        self.assertEqual(v.elements, [1, 2, 3])

    def test_from_list_of_floats(self):
        v = Vec([1.0, 2.5, -3.7])
        self.assertEqual(v.elements, [1.0, 2.5, -3.7])

    def test_from_tuple(self):
        """Any iterable should work, not just lists."""
        v = Vec((4, 5, 6))
        self.assertEqual(v.elements, [4, 5, 6])

    def test_empty_default(self):
        v = Vec()
        self.assertEqual(v.elements, [])

    def test_empty_explicit_none(self):
        v = Vec(None)
        self.assertEqual(v.elements, [])

    def test_invalid_entry_raises_typeerror(self):
        with self.assertRaises(TypeError):
            Vec([1, "two", 3])

    def test_invalid_entry_bool_accepted(self):
        # bool is a subclass of int in Python; Vec should accept it
        # (this documents the behaviour rather than asserting it wrong)
        v = Vec([True, False])
        self.assertEqual(len(v), 2)


# ===========================================================================
# 2. __len__  and  __repr__
# ===========================================================================

class TestBuiltins(unittest.TestCase):

    def test_len(self):
        self.assertEqual(len(Vec([1, 2, 3, 4])), 4)

    def test_len_empty(self):
        self.assertEqual(len(Vec()), 0)

    def test_repr_matches_list_repr(self):
        v = Vec([1, 2, 3])
        self.assertEqual(repr(v), repr([1, 2, 3]))

    def test_repr_empty(self):
        self.assertEqual(repr(Vec()), repr([]))


# ===========================================================================
# 3. Factory methods  (zeros, ones, uniform)
# ===========================================================================

class TestFactoryMethods(unittest.TestCase):

    # --- zeros ---

    def test_zeros_length(self):
        self.assertEqual(len(Vec.zeros(5)), 5)

    def test_zeros_all_zero(self):
        v = Vec.zeros(6)
        self.assertTrue(all(x == 0 for x in v.elements))

    def test_zeros_returns_vec(self):
        self.assertIsInstance(Vec.zeros(3), Vec)

    def test_zeros_invalid_n_raises(self):
        with self.assertRaises(ValueError):
            Vec.zeros(0)
        with self.assertRaises(ValueError):
            Vec.zeros(-1)

    # --- ones ---

    def test_ones_length(self):
        self.assertEqual(len(Vec.ones(4)), 4)

    def test_ones_all_one(self):
        v = Vec.ones(5)
        self.assertTrue(all(x == 1 for x in v.elements))

    def test_ones_returns_vec(self):
        self.assertIsInstance(Vec.ones(3), Vec)

    def test_ones_invalid_n_raises(self):
        with self.assertRaises(ValueError):
            Vec.ones(0)

    # --- uniform ---

    def test_uniform_length(self):
        self.assertEqual(len(Vec.uniform(10)), 10)

    def test_uniform_entries_in_range(self):
        v = Vec.uniform(100)
        self.assertTrue(all(0.0 <= x < 1.0 for x in v.elements))

    def test_uniform_returns_vec(self):
        self.assertIsInstance(Vec.uniform(3), Vec)

    def test_uniform_invalid_n_raises(self):
        with self.assertRaises(ValueError):
            Vec.uniform(0)

    def test_uniform_not_all_same(self):
        """100 random draws should not all be identical (astronomically unlikely)."""
        v = Vec.uniform(100)
        self.assertGreater(len(set(v.elements)), 1)


# ===========================================================================
# 4. __add__
# ===========================================================================

class TestAdd(unittest.TestCase):

    def test_basic_add(self):
        result = Vec([1, 2, 3]) + Vec([4, 5, 6])
        self.assertEqual(result.elements, [5, 7, 9])

    def test_add_with_floats(self):
        result = Vec([0.1, 0.2]) + Vec([0.3, 0.4])
        self.assertTrue(vec_approx(result, Vec([0.4, 0.6]), tol=1e-5))

    def test_add_commutativity(self):
        a = Vec([1.0, 2.0, 3.0])
        b = Vec([4.0, 5.0, 6.0])
        self.assertTrue(vec_approx(a + b, b + a))

    def test_add_associativity(self):
        a = Vec([1.0, 0.0])
        b = Vec([0.0, 1.0])
        c = Vec([1.0, 1.0])
        self.assertTrue(vec_approx((a + b) + c, a + (b + c)))

    def test_add_zero_vector_is_identity(self):
        v = Vec([3.0, -1.0, 5.0])
        z = Vec.zeros(3)
        self.assertTrue(vec_approx(v + z, v))

    def test_add_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            Vec([1, 2]) + 5

    def test_add_dimension_mismatch_raises(self):
        with self.assertRaises(TypeError):
            Vec([1, 2]) + Vec([1, 2, 3])

    def test_add_returns_new_vec(self):
        """__add__ must not mutate either operand."""
        a = Vec([1.0, 2.0])
        b = Vec([3.0, 4.0])
        _ = a + b
        self.assertEqual(a.elements, [1.0, 2.0])
        self.assertEqual(b.elements, [3.0, 4.0])


# ===========================================================================
# 5. __radd__
# ===========================================================================

class TestRadd(unittest.TestCase):

    def test_radd_with_zero_seed(self):
        """sum() starts accumulating from 0; __radd__ must handle that."""
        v = Vec([1.0, 2.0, 3.0])
        result = sum([v])
        self.assertTrue(vec_approx(result, v))

    def test_sum_multiple_vecs(self):
        v1 = Vec([1.0, 0.0])
        v2 = Vec([0.0, 1.0])
        v3 = Vec([2.0, 3.0])
        result = sum([v1, v2, v3])
        self.assertTrue(vec_approx(result, Vec([3.0, 4.0])))

    def test_radd_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ = 5 + Vec([1, 2])


# ===========================================================================
# 6. __iadd__  (+=)
# ===========================================================================

class TestIadd(unittest.TestCase):

    def test_iadd_basic(self):
        v = Vec([1.0, 2.0, 3.0])
        v += Vec([4.0, 5.0, 6.0])
        self.assertEqual(v.elements, [5.0, 7.0, 9.0])

    def test_iadd_mutates_in_place(self):
        """After +=, the same object should be updated."""
        v = Vec([1.0, 2.0])
        original_id = id(v)
        v += Vec([1.0, 1.0])
        self.assertEqual(id(v), original_id)

    def test_iadd_wrong_type_raises(self):
        v = Vec([1.0, 2.0])
        with self.assertRaises(TypeError):
            v += 3

    def test_iadd_dimension_mismatch_raises(self):
        v = Vec([1.0, 2.0])
        with self.assertRaises(TypeError):
            v += Vec([1.0, 2.0, 3.0])


# ===========================================================================
# 7. __sub__
# ===========================================================================

class TestSub(unittest.TestCase):

    def test_basic_sub(self):
        result = Vec([5, 7, 9]) - Vec([1, 2, 3])
        self.assertEqual(result.elements, [4, 5, 6])

    def test_sub_self_is_zero(self):
        v = Vec([3.0, -1.0, 5.0])
        z = v - v
        self.assertTrue(all(approx(x, 0.0) for x in z.elements))

    def test_sub_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            Vec([1, 2]) - 3

    def test_sub_dimension_mismatch_raises(self):
        with self.assertRaises(TypeError):
            Vec([1, 2]) - Vec([1, 2, 3])

    def test_sub_returns_new_vec(self):
        a = Vec([5.0, 6.0])
        b = Vec([1.0, 2.0])
        _ = a - b
        self.assertEqual(a.elements, [5.0, 6.0])


# ===========================================================================
# 8. __neg__
# ===========================================================================

class TestNeg(unittest.TestCase):

    def test_neg_basic(self):
        result = -Vec([1, -2, 3])
        self.assertEqual(result.elements, [-1, 2, -3])

    def test_neg_zero_vector(self):
        z = Vec.zeros(4)
        self.assertTrue(vec_approx(-z, z))

    def test_double_neg_is_identity(self):
        v = Vec([3.0, -1.0, 4.0])
        self.assertTrue(vec_approx(-(-v), v))

    def test_neg_returns_new_vec(self):
        v = Vec([1.0, 2.0])
        _ = -v
        self.assertEqual(v.elements, [1.0, 2.0])


# ===========================================================================
# 9. __rmul__  (scalar * Vec)
# ===========================================================================

class TestRmul(unittest.TestCase):

    def test_rmul_int_scalar(self):
        result = 3 * Vec([1, 2, 3])
        self.assertEqual(result.elements, [3, 6, 9])

    def test_rmul_float_scalar(self):
        result = 2.0 * Vec([1.0, 2.0])
        self.assertTrue(vec_approx(result, Vec([2.0, 4.0])))

    def test_rmul_zero_scalar_gives_zero_vec(self):
        result = 0 * Vec([5.0, -3.0, 1.0])
        self.assertTrue(all(approx(x, 0.0) for x in result.elements))

    def test_rmul_negative_scalar(self):
        result = -1 * Vec([1.0, -2.0, 3.0])
        self.assertTrue(vec_approx(result, Vec([-1.0, 2.0, -3.0])))

    def test_rmul_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            "a" * Vec([1, 2])

    def test_rmul_returns_new_vec(self):
        v = Vec([1.0, 2.0])
        _ = 5 * v
        self.assertEqual(v.elements, [1.0, 2.0])


# ===========================================================================
# 10. __imul__  (*=)
# ===========================================================================

class TestImul(unittest.TestCase):

    def test_imul_basic(self):
        v = Vec([1.0, 2.0, 3.0])
        v *= 4
        self.assertEqual(v.elements, [4.0, 8.0, 12.0])

    def test_imul_mutates_in_place(self):
        v = Vec([1.0, 2.0])
        original_id = id(v)
        v *= 3
        self.assertEqual(id(v), original_id)

    def test_imul_wrong_type_raises(self):
        v = Vec([1.0, 2.0])
        with self.assertRaises(TypeError):
            v *= "x"


# ===========================================================================
# 11. norm
# ===========================================================================

class TestNorm(unittest.TestCase):

    def test_norm_345_triangle(self):
        """3-4-5 right triangle: norm([3, 4]) == 5."""
        self.assertAlmostEqual(Vec([3, 4]).norm(), 5.0, places=9)

    def test_norm_unit_vector(self):
        self.assertAlmostEqual(Vec([1, 0, 0]).norm(), 1.0, places=9)

    def test_norm_zero_vector(self):
        self.assertAlmostEqual(Vec.zeros(5).norm(), 0.0, places=9)

    def test_norm_ones_vector(self):
        """‖1_n‖ = sqrt(n)."""
        n = 9
        self.assertAlmostEqual(Vec.ones(n).norm(), math.sqrt(n), places=9)

    def test_norm_nonnegative(self):
        self.assertGreaterEqual(Vec([-3.0, 4.0]).norm(), 0.0)

    def test_norm_lecture_example(self):
        """Lecture: ‖(10,-2,8,23,6,75)‖ ≈ 79.7371."""
        x = Vec([10, -2, 8, 23, 6, 75])
        self.assertAlmostEqual(x.norm(), 79.7371, places=3)

    def test_norm_homogeneity(self):
        """‖a·x‖ = |a| · ‖x‖  (nonnegative homogeneity)."""
        v = Vec([1.0, 2.0, 3.0])
        a = -4.0
        self.assertAlmostEqual((a * v).norm(), abs(a) * v.norm(), places=9)

    def test_norm_empty_vector_is_zero(self):
        self.assertAlmostEqual(Vec().norm(), 0.0, places=9)


# ===========================================================================
# 12. mean
# ===========================================================================

class TestMean(unittest.TestCase):

    def test_mean_basic(self):
        """Lecture example: mean of (10,-2,8,23,6,75) == 20."""
        self.assertTrue(approx(Vec([10, -2, 8, 23, 6, 75]).mean(), 20.0))

    def test_mean_single_entry(self):
        self.assertTrue(approx(Vec([7.0]).mean(), 7.0))

    def test_mean_all_equal_entries(self):
        """If every entry equals c, the mean equals c."""
        c = 5.0
        self.assertTrue(approx(Vec([c, c, c, c]).mean(), c))

    def test_mean_definition(self):
        """mean == sum / n."""
        entries = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        self.assertTrue(approx(Vec(entries).mean(), sum(entries) / len(entries)))

    def test_mean_negative_entries(self):
        self.assertTrue(approx(Vec([-3.0, -1.0, 1.0, 3.0]).mean(), 0.0))

    def test_mean_shift_by_constant(self):
        """Adding constant c to every entry shifts mean by c."""
        x = Vec([1.0, 2.0, 3.0, 4.0, 5.0])
        c = 10.0
        shifted = Vec([v + c for v in x.elements])
        self.assertTrue(approx(shifted.mean(), x.mean() + c))

    def test_mean_scale_by_scalar(self):
        """Multiplying every entry by a scales mean by a."""
        x = Vec([2.0, 4.0, 6.0])
        a = 3.0
        self.assertTrue(approx((a * x).mean(), a * x.mean()))

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            Vec().mean()


# ===========================================================================
# 13. demean
# ===========================================================================

class TestDemean(unittest.TestCase):

    def test_demean_basic_values(self):
        """Lecture example: demean of (10,-2,8,23,6,75) == (-10,-22,-12,3,-14,55)."""
        x = Vec([10, -2, 8, 23, 6, 75])
        expected = Vec([-10.0, -22.0, -12.0, 3.0, -14.0, 55.0])
        self.assertTrue(vec_approx(x.demean(), expected, tol=1e-6))

    def test_demean_mean_is_zero(self):
        """Key property: mean of the de-meaned vector is always zero."""
        x = Vec([10, -2, 8, 23, 6, 75])
        self.assertTrue(approx(x.demean().mean(), 0.0, tol=1e-9))

    def test_demean_mean_is_zero_general(self):
        x = Vec([3.5, 7.1, -2.4, 9.0, 0.0, 4.8])
        self.assertTrue(approx(x.demean().mean(), 0.0, tol=1e-9))

    def test_demean_constant_vector_is_zero_vector(self):
        """Quiz (lecture): if all entries equal c, de-meaned vector is 0."""
        c = 42.0
        x = Vec([c, c, c, c, c])
        self.assertTrue(all(approx(v, 0.0) for v in x.demean().elements))

    def test_demean_does_not_mutate_original(self):
        entries = [1.0, 2.0, 3.0, 4.0]
        x = Vec(entries)
        _ = x.demean()
        self.assertEqual(x.elements, entries)

    def test_demean_single_element(self):
        self.assertTrue(approx(Vec([99.0]).demean().elements[0], 0.0))

    def test_demean_sum_is_zero(self):
        """sum of de-meaned entries == 0  (equivalent to mean == 0)."""
        x = Vec([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])
        self.assertTrue(approx(sum(x.demean().elements), 0.0, tol=1e-9))

    def test_demean_shift_invariance(self):
        """Adding a constant to every entry does not change the de-meaned vector."""
        x = Vec([1.0, 3.0, 5.0, 7.0])
        shifted = Vec([v + 100.0 for v in x.elements])
        self.assertTrue(vec_approx(x.demean(), shifted.demean(), tol=1e-9))


# ===========================================================================
# 14. std
# ===========================================================================

class TestStd(unittest.TestCase):

    def test_std_lecture_example(self):
        """Lecture: std of (10,-2,8,23,6,75) ≈ 25.686."""
        x = Vec([10, -2, 8, 23, 6, 75])
        self.assertAlmostEqual(x.std(), 25.686, places=2)

    def test_std_good_sample_from_lecture(self):
        """Lecture: std of (10,2,9,18,6,15,12,7,11) == 4.522."""
        x = Vec([10, 2, 9, 18, 6, 15, 12, 7, 11])
        self.assertAlmostEqual(x.std(), 4.522, places=2)

    def test_std_equals_norm_demeaned_over_sqrt_n(self):
        """std(x) == ‖x̃‖ / sqrt(n)  — the lecture formula."""
        x = Vec([10, -2, 8, 23, 6, 75])
        dm = x.demean()
        norm_dm = math.sqrt(sum(v * v for v in dm.elements))
        self.assertAlmostEqual(x.std(), norm_dm / math.sqrt(len(x)), places=9)

    def test_std_constant_vector_is_zero(self):
        """Lecture: std is zero iff all entries are equal."""
        self.assertAlmostEqual(Vec([7.0, 7.0, 7.0, 7.0]).std(), 0.0, places=9)

    def test_std_nonnegative(self):
        self.assertGreaterEqual(Vec([-5.0, 3.0, 0.0, 1.0, -2.0]).std(), 0.0)

    def test_std_shift_invariance(self):
        """Lecture property: adding a constant does NOT change std."""
        x = Vec([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        shifted = Vec([v + 100.0 for v in x.elements])
        self.assertAlmostEqual(shifted.std(), x.std(), places=9)

    def test_std_scalar_multiplication(self):
        """Lecture property: std(a·x) == |a| · std(x)."""
        x = Vec([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        a = -3.0
        self.assertAlmostEqual((a * x).std(), abs(a) * x.std(), places=9)

    def test_std_single_element_is_zero(self):
        self.assertAlmostEqual(Vec([42.0]).std(), 0.0, places=9)

    def test_std_empty_raises(self):
        with self.assertRaises(ValueError):
            Vec().std()

    def test_std_consistency_with_demean(self):
        """Manual std from demean() must match Vec.std()."""
        x = Vec([1.0, 2.0, 3.0, 4.0, 5.0])
        dm = x.demean()
        manual = math.sqrt(sum(v * v for v in dm.elements) / len(x))
        self.assertAlmostEqual(x.std(), manual, places=9)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
