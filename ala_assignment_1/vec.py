
import sys
from typing import Self


"""
A custom vector class implementation for educational purposes.
"""

class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = []
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)])


    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        #
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)
        #
        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")
        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)])

    def __neg__(self) -> Self:
        return Vec([-x for x in self.elements])

    def __radd__(self, other):
        # Python's built-in sum() starts with integer 0; treat it as the
        # additive identity so that sum([v1, v2, ...]) works correctly.
        if other == 0:
            return self
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        if len(self.elements) != len(other):
            raise TypeError("Vectors must have the same dimension for addition")
        for i, y in enumerate(other.elements):
            self.elements[i] = round(self.elements[i] + y, 5)
        return self

    # return a vector of @n zeroes. precondition: @n > 0
    @staticmethod
    def zeros(n: int) -> "Vec":
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        return Vec([0] * n)

    # return a vector of @n ones. precondition: @n > 0
    @staticmethod
    def ones(n: int) -> "Vec":
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        return Vec([1] * n)

    # return a vector of @n uniformly distributed numbers in [0, 1). precondition: @n > 0
    @staticmethod
    def uniform(n: int) -> "Vec":
        import random
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        return Vec([random.random() for _ in range(n)])

    # Calculates the Euclidean norm (L2 norm) of the vector.
    # sqrt(e[0]^2 + e[1]^2 + e[2]^2 + ... + e[n-1]^2)
    def norm(self) -> float:
        import math
        return math.sqrt(sum(x * x for x in self.elements))

    def mean(self) -> float:
        """Return the mean (average) of the vector's entries.

        mean(x) = (x[0] + x[1] + ... + x[n-1]) / n
        """
        if len(self.elements) == 0:
            raise ValueError("mean is undefined for an empty vector")
        return sum(self.elements) / len(self.elements)

    def demean(self) -> "Vec":
        """Return a new vector that is the de-meaned version of this vector.

        The de-meaned vector x̃ is obtained by subtracting the mean from
        every entry:  x̃[i] = x[i] - mean(x)

        A key property: the mean of the de-meaned vector is always zero.
        """
        m = self.mean()
        return Vec([round(x - m, 10) for x in self.elements])

    def std(self) -> float:
        """Return the standard deviation of the vector's entries.

        std(x) = norm(x̃) / sqrt(n)
               = sqrt( (1/n) * sum( (x[i] - mean(x))^2 ) )

        Uses demean() to obtain the deviations from the mean.
        """
        if len(self.elements) == 0:
            raise ValueError("std is undefined for an empty vector")
        n = len(self.elements)
        demeaned = self.demean()
        sum_sq = sum(v * v for v in demeaned.elements)
        return (sum_sq / n) ** 0.5


"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.
(5) Test this implementation by importing the class in a sepatate python script.

(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements?
(7) Measure the performance on your machine. Check it on colab.

(8) use numpy and compare the performance.
"""


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    v1 = Vec([0, 1, 1.03])
    print("v1          :", v1)

    v3 = 2.2 * v1
    v3 *= 5
    print("2.2*v1 *=5  :", v3)

    print("v1 + v3     :", v1 + v3)
    print("v1 - v3     :", v1 - v3)
    print("-v1         :", -v1)

    # __iadd__
    v4 = Vec([1.0, 0.0, 0.0])
    v4 += Vec([0.0, 1.0, 0.0])
    print("v4 += ...   :", v4)

    # __radd__ — sum() seeds with 0, so this exercises __radd__
    print("sum([v1,v1]):", sum([v1, v1]))

    # factory methods
    print("zeros(4)    :", Vec.zeros(4))
    print("ones(4)     :", Vec.ones(4))
    print("uniform(4)  :", Vec.uniform(4))

    # norm, mean, demean, std
    x = Vec([10, -2, 8, 23, 6, 75])
    print("norm        :", round(x.norm(), 4))
    print("mean        :", x.mean())
    print("demean      :", x.demean())
    print("std         :", round(x.std(), 3))
