from tools.helper import euklidian_distance, normalized_compression_distance, equal_elements_distance
import math

from bz2 import compress


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestNovelty:
    def test_euklidian_distance(self):
        assert 1 == euklidian_distance([0, 0, 0], [1, 0, 0])
        assert math.sqrt(3) == euklidian_distance([0, 0, 0], [1, 1, 1])
        assert math.sqrt(3) == euklidian_distance([0, 0, 0], [1, 1, 1, 1, 2, 3, 4, 5, 3, 6, 7, 8])

    def test_equal_elements_distance(self):
        assert 0 == equal_elements_distance([1, 1], [1, 1])
        assert 2 == equal_elements_distance([1, 2], [3, 4])

        # position is important
        assert 2 == equal_elements_distance([1, 2], [2, 1])

        # when 2nd is longer, it will be cut off
        assert 0 == equal_elements_distance([1, 1], [1, 1, 1, 1, 1, 1])

        # when 1st is longer, it will count towards novelty
        assert 4 == equal_elements_distance([1, 1, 1, 1, 1, 1], [1, 1])

    def test_normalized_compression_distance(self):
        a = [0, 1] * 1000
        b = [1, 0] * 1000
        c = [1, 4, 6, 46, 8, 2, 5, 4, 4, 0, 4, 8, 46, 52, 4, 5, 7, 8, 8, 4] * 100
        d = [1, 0] * 10

        # test symmetry
        assert isclose(normalized_compression_distance(a, c), normalized_compression_distance(c, a))
        assert isclose(normalized_compression_distance(a, b), normalized_compression_distance(b, a))

        # test plausibility
        assert normalized_compression_distance(a, b) < normalized_compression_distance(a, c)
        assert normalized_compression_distance(a, d) < normalized_compression_distance(a, c)

        # test uneven lengths and plausibility
        assert normalized_compression_distance(a, d) <= normalized_compression_distance(a, b)
        assert normalized_compression_distance(b, d) < normalized_compression_distance(a, d)
        assert normalized_compression_distance(c, c * 10) < normalized_compression_distance(a, c)
        assert normalized_compression_distance(c, c) < normalized_compression_distance(c, c * 10)

        # test precomputed values
        assert isclose(
            normalized_compression_distance(a=a, b=b),
            normalized_compression_distance(a=a, b=b,
                                            a_len=len(compress(bytearray(a), 1)),
                                            b_len=len(compress(bytearray(b), 1))
                                            ))
