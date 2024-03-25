'''
Unit tests for a few specific parts of the cost estimation.
'''


from estimate_costs import _memoize, find_recursion_size, get_eval_pts, compute_sum_size
import unittest as ut


# for the below
def dummy_test_fn(a, b, c, kw1, kw2):
    return [str(v) for v in [a, b, c, kw1, kw2]]

@_memoize
def memoized_test_fn(*args, **kwargs):
    return dummy_test_fn(*args, **kwargs)

class Memoize(ut.TestCase):
    def test_memoization(self):
        '''
        ensure the function output is correct for various
        combinations of arguments, when you run it multiple
        times
        '''
        test_args = [
            ('??', 12, 402),
            ('new', 1, 2),
            ('new', 12, 402),
            ('??', 12, 4),
            ('x', 1, 402),
        ]
        for a, b, c in test_args+test_args:
            for kw1 in [True, False]:
                for kw2 in [False, True]:
                    result = memoized_test_fn(a, b, c, kw1=kw1, kw2=kw2)
                    correct = dummy_test_fn(a, b, c, kw1=kw1, kw2=kw2)
                    self.assertEqual(result, correct)

                    # make sure we can't modify the result stored in memoization
                    result.append('test')


class Recursion(ut.TestCase):
    def test_find_size(self):
        test_cases = [
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 2),
            (11, 3),
            (11, 4),
        ]
        for n, k in test_cases:
            with self.subTest(n=n, k=k):
                top_reg_size, reg_size = find_recursion_size(n, k)
                self.assertEqual(top_reg_size + (k-1)*reg_size, n)

    def test_get_eval_pts(self):
        test_cases = [
            ('cq', 2, [(0, False), (0, True), (-1, True)]),
            ('poq', 2, [(0, False), (0, True), (-1, True), (1, True)]),
            ('cq', 3, [(0, False), (0, True), (-1, True), (1, True), (-2, True)]),
            ('poq', 3, [(0, False), (0, True), (-1, True), (1, True), (-2, True), (2, True), (-2, False)]),
            ('cq', 4, [(0, False), (0, True), (-1, True), (1, True), (-2, True), (2, True), (-2, False)]),
            ('poq', 4, [(0, False), (0, True), (-1, True), (1, True), (-2, True), (2, True), (-2, False), (2, False), (-4, True), (4, True)]),
        ]

        for op, k, correct in test_cases:
            with self.subTest(op=op, k=k):
                for i, (check, correct) in enumerate(zip(get_eval_pts(op, k), correct, strict=True)):
                    with self.subTest(i=i):
                        self.assertEqual(check, correct)

    @staticmethod
    def _explicit_sum_size(wl, inv, k, top_reg_size, reg_size):
        top_max = 2**top_reg_size - 1
        other_max = 2**reg_size - 1

        if wl < 0:
            # first try making the even ones positive
            tot1 = 0
            for i in range(0, k, 2):
                if (inv and i==0) or (not inv and i==(k-1)):
                    reg_max = top_max
                else:
                    reg_max = other_max

                tot1 += wl**i * reg_max

            # then try the odd ones
            tot2 = 0
            for i in range(1, k, 2):
                if not inv and i==(k-1):
                    reg_max = top_max
                else:
                    reg_max = other_max

                tot2 += wl**i * reg_max

            tot = max(abs(tot1), abs(tot2))

        else:
            # include all
            tot = 0
            for i in range(k):
                if (inv and i==0) or (not inv and i==(k-1)):
                    reg_max = top_max
                else:
                    reg_max = other_max

                tot += wl**i * reg_max

        return tot.bit_length()

    def test_sum_size(self):
        for wl in [1, -1, 2, -2, 4, -4]:
            for inv in [True, False]:
                for k in [2, 4, 5, 6]:
                    for top_reg_size in [10, 12, 14, 16]:
                        reg_size = 10

                        with self.subTest(
                                wl=wl,
                                inv=inv,
                                k=k,
                                top_reg_size=top_reg_size,
                                reg_size=reg_size
                        ):
                            self.assertEqual(
                                compute_sum_size(wl, inv, k, top_reg_size, reg_size),
                                self._explicit_sum_size(wl, inv, k, top_reg_size, reg_size)
                            )


if __name__ == '__main__':
    ut.main()
