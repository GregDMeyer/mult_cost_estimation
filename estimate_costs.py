'''
This script estimates the resource costs, in terms of gates and qubits, required for a particular
implementation of fast Fourier-based quantum multiplication.
'''

from collections import defaultdict
from math import ceil, floor, log2
from argparse import ArgumentParser, BooleanOptionalAction
from functools import wraps
from itertools import islice


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('op_type', choices=['cq', 'poq_fast', 'poq_balanced', 'poq_narrow'], help='which type of multiplication to perform. cq=classical quantum; poq=proof of quantumness (x^2 mod N)')
    parser.add_argument('--non-modular', action='store_true',
                        help='compute the full product, instead of the product mod N')
    parser.add_argument('--gradient-qft', action='store_true',
                        help='use phase gradient state to implement QFT')

    return parser.parse_args()


def main():
    args = parse_args()

    headings = ['size', 'toffolis', 'phase rots.', 'measmts.', 'other gates', 'max_k']

    if args.op_type == 'cq':
        sizes = [2**i for i in range(6, 13)]
        headings.append('ancillas')
    elif 'poq' in args.op_type:
        sizes = [128, 400, 829, 1024]
        headings.append('total qubits')
    else:
        raise ValueError('unrecognized op_type')

    gate_types = set()

    print(', '.join(headings))
    for size in sizes:

        # square: both inputs are the same register
        if 'cq' in args.op_type:
            square = False
        elif 'poq' in args.op_type:
            square = True
        else:
            raise ValueError('unknown op type')

        double_z = args.non_modular

        # narrow has a different structure
        if args.op_type == 'poq_narrow':
            result = estimate_poq_narrow_cost(size, square=square, double_z=double_z)
        else:
            result = estimate_cost(args.op_type, size, square=square, double_z=double_z)

        # cq requires initial and final qft. we use 1E-12 precision, which is
        # probably overkill but it's worth being conservative
        if 'cq' in args.op_type:
            result += 2*qft_cost(size, precision=1E-12, use_phase_gradient=args.gradient_qft)

        # for poq we can do Hadamards at the start and semiclassical qft at the end
        # since we are measuring immediately. very cheap!
        else:
            if args.gradient_qft:
                raise ValueError('does not make sense to use gradient QFT for PoQ')
            result += CategoryCounter({'H': size, 'phase': size})

        other_gates = result.total_gates()
        other_gate_exclude = []  # don't include in "other"

        tof_gates = result.count_gates(['toffoli'])
        other_gate_exclude.append('toffoli')
        other_gates -= tof_gates

        phase_gates = result.phase_rotations()
        other_gate_exclude += [k for k in result.counts if 'phase' in k]
        other_gates -= phase_gates

        measmts = result.count_gates(['measurement'])
        other_gate_exclude.append('measurement')
        other_gates -= measmts

        gate_types.update(result.counts)

        # count total qubits for poq
        if args.op_type == 'poq_narrow':
            result.ancillas += 3*size // 2
        elif args.op_type in ('poq_balanced', 'poq_fast'):
            result.ancillas += 2*size
        elif 'cq' not in args.op_type:
            raise ValueError('unknown op_type')

        items = [
            size,
            tof_gates,
            phase_gates,
            measmts,
            other_gates,
            result.max_k,
            result.ancillas,
        ]
        print(
            *[str(item).rjust(len(heading)) for item, heading in zip(items, headings)],
            sep=', '
        )

    print()
    print('Other gates include:')
    for gate_type in gate_types:
        if gate_type in other_gate_exclude:
            continue
        print('  '+gate_type)


def _memoize(func):
    '''
    Memoize the result of the function, so when it is called
    with the same arguments there is no need to run it again
    '''
    memo_dict = {}

    @wraps(func)
    def new_func(*args, **kwargs):
        key = (tuple(args), tuple(sorted(kwargs.items(), key=lambda x: x[0])))
        if key not in memo_dict:
            result = func(*args, **kwargs)
            memo_dict[key] = result
        return memo_dict[key].copy()

    return new_func


# optimize _estimate_cost by iterating over k
# the size of the search tree is exponential of a log
# so it is not excessive
@_memoize
def estimate_cost(op, n, **kwargs):
    max_k = 12  # arbitrary but empirically seems big enough
    if 'poq' in op:
        min_k = 3
    elif op == 'cq':
        min_k = 2
    else:
        raise ValueError('unknown value for op')

    # start with base case
    best_cost = base_case(op, n, **kwargs)
    for k in range(min_k, max_k):

        cost = _estimate_cost(op, n, k, **kwargs)
        if cost is None:
            # the recursion failed, e.g. too small
            continue

        if cost_function(best_cost) > cost_function(cost):
            best_cost = cost

    return best_cost


def cost_function(gate_counter):
    # could do something more complicated, but it doesn't seem to change the result much
    return gate_counter.count_gates(['toffoli']) + 4*gate_counter.count_gates(['cphase'])


def _estimate_cost(op, n, k, square, double_z):
    gates = CategoryCounter()

    registers = 2
    if 'poq' in op and not square:
        registers += 1  # two input registers, x and y
    if double_z:
        registers += 1  # low and high halves of output are separate registers for our purposes

    top_reg_size, reg_size = find_recursion_size(n, k)

    if reg_size <= 1:
        # we have recursed too far!
        return None

    for wl, inv in get_eval_pts(op, k):
        sub_gates = CategoryCounter()

        # if wl=0, we don't need to do any additions
        if wl == 0:
            if inv:
                # wl = infty
                recurse_n = top_reg_size
            else:
                recurse_n = reg_size
            ancillas = 0

        # need the full machinery of additions etc
        else:
            recurse_n = compute_sum_size(wl, inv, k, top_reg_size, reg_size)

            if recurse_n >= n:
                # the recursion isn't making the problem smaller!
                return None

            # all positive wl are preceded by negative of the same---only one extra addition needed!
            if wl > 0:
                n_adds = 1
            else:
                if op == 'poq_fast':
                    n_adds = k-1   # uncomputation via measurement
                    ancillas = recurse_n*registers
                    if k > 3:
                        # store intermediate sum of negative coeffs
                        ancillas += recurse_n*registers
                    sub_gates += CategoryCounter({'measurement': ancillas})

                else:
                    n_adds = 2*(k-1)  # computation and uncomputation

                    # store linear combination in top register to reduce ancillas
                    ancillas = recurse_n - top_reg_size
                    if k > 3:
                        # store intermediate sum, overwriting x_1
                        ancillas += recurse_n - reg_size

                # sign bit
                ancillas += 1

            # a slight overestimate because some sums are not fully recurse_n bits
            sub_gates += n_adds*registers*count_add(recurse_n)

        # the recursive call!
        sub_gates += estimate_cost(op, recurse_n, square=square, double_z=double_z)
        sub_gates.ancillas += ancillas
        gates += sub_gates

    gates.include_k(k)

    return gates


def find_recursion_size(n, k):
    '''
    Given a product size and k, figure out the size of the
    "subregisters" that the values will be divided into
    '''
    # one could play with this and see how it affects things
    reg_size = n//k
    top_reg_size = n-(reg_size*(k-1))
    return top_reg_size, reg_size


def get_eval_pts(op, k):
    '''
    creates an iterator that yields the points w_\ell at which the
    polynomials are evaluated
    '''
    if 'poq' in op:
        q_factor = 3
    elif op == 'cq':
        q_factor = 2
    else:
        raise ValueError('unknown op')

    q = q_factor*(k-1)+1

    def pt_generator():
        yield (0, False)
        yield (0, True)

        x = 1
        while True:
            # note: the code in _estimate_cost relies on
            # the negative points occurring before the
            # positive ones!
            yield (-x, True)
            yield (x, True)
            if x>1:
                yield (-x, False)
                yield (x, False)
            x *= 2

    return islice(pt_generator(), q)


def compute_sum_size(wl, inv, k, top_reg_size, reg_size):
    '''
    compute how many overflow bits will occur in the sum
    '''
    coeffs = [abs(wl)**i for i in range(k)]
    if inv:  # largest coeff is on smallest register
        coeffs = coeffs[::-1]

    reg_values = [2**reg_size - 1]*(k-1) + [2**top_reg_size - 1]

    if wl < 0:
        # if wl is negative, the result is maximized when either the even or odd values are maximized
        max_val = max(
            sum(c*x for i, (c, x) in enumerate(zip(coeffs, reg_values)) if i%2 == 0),
            sum(c*x for i, (c, x) in enumerate(zip(coeffs, reg_values)) if i%2 == 1)
        )
    else:
        # otherwise it's just when everything is maximum
        max_val = sum(c*x for c, x in zip(coeffs, reg_values))

    return max_val.bit_length()


def estimate_poq_narrow_cost(n, square, double_z):
    '''
    Estimate the cost of running the circuit in which the mult
    is computed as x_1^2 + 2*x_0*x_1 + x_0^2
    '''
    sub_n = (n+1)//2

    # x_0^2 and x_1^2
    result = 2*estimate_cost('poq_balanced', sub_n, square=square, double_z=double_z)

    # x_0*x_1
    result += estimate_cost('poq_balanced', sub_n, square=False, double_z=double_z)

    # double it, because we do it once for each half of z
    return 2*result


def base_case(op, n, square, double_z):
    if 'cq' in op:
        return CategoryCounter({'cphase': n**2*(2 if double_z else 1)})

    # otherwise we are doing poq
    return count_small_poq_multiplier(n, square, double_z)


def count_small_poq_multiplier(n, square, double_z):
    sum_dict = defaultdict(lambda: 0)

    half_adders = 0
    full_adders = 0
    cphase_arrays = 0
    toffolis = 0
    measurements = 0

    for m in range(2*n-1):
        for i in range(n):
            j = m-i
            if not 0 <= j < n:
                continue

            if square and j < i:
                continue

            if not (i == j and square):
                toffolis += 1

            if i != j and square:  # count it twice
                sum_dict[m+1] += 1
            else:
                sum_dict[m] += 1

        # apply full adders to reduce the number of bits stored at this level
        while sum_dict[m] > 2:
            full_adders += 1
            measurements += 2  # measuring away the qubits we are dropping
            sum_dict[m] -= 2
            sum_dict[m+1] += 1

        # at this point there are 1 or 2 bits at level m
        # if still 2, use a half adder to reduce to 1
        if sum_dict[m] == 2:
            half_adders += 1
            measurements += 1
            sum_dict[m] -= 1
            sum_dict[m+1] += 1

        # number of passes across the bits of z
        cphase_arrays += 1

        # measure away our final bit
        measurements += 1

    if double_z:
        # effectively passing over z twice as many times
        cphase_arrays *= 2

    # note: there is probably a more optimal way to implement a classical adder with measurement-based uncomputation
    # we use the following as a conservative estimate
    return CategoryCounter({
        'toffoli': toffolis + 2*full_adders + half_adders,
        'cnot': 2*full_adders + half_adders,
        'measurement': measurements,
        'cphase': n*cphase_arrays
    })


def count_add(n):
    '''
    Cost of addition via Cuccaro's quantum adder
    '''
    if n <= 1:
        raise ValueError('n too small for cuccaro adder')

    return CategoryCounter({
        'toffoli': 2*n - 1,
        'cnot': 5*n - 3,
        'X': 2*n - 4
    })


def qft_cost(n, precision, use_phase_gradient):
    '''
    Cost of a QFT in which all rotations smaller than "precision" are dropped.
    use_phase_gradient: see https://algassert.com/post/1620
    '''
    log_precision = -int(floor(log2(precision)))
    cutoff = min(log_precision, n)

    if use_phase_gradient:
        # set up phase gradient
        rtn = CategoryCounter({'phase': cutoff})

        # initial rotations up to cutoff
        for i in range(2, cutoff):
            # CNOTs putting value into ancilla register, and uncomputing after
            rtn += CategoryCounter({'cnot': 2*i})
            # apply phase rotation using gradient state
            rtn += count_add(i)

        # remaining rotations
        for i in range(cutoff, n):
            rtn += CategoryCounter({'cnot': 2*cutoff})
            rtn += count_add(cutoff)

        # cutoff worth for phase gradient, another cutoff worth for adder ancillas
        rtn.ancillas = 2*cutoff
    else:
        gates = (n-cutoff)*cutoff
        gates += cutoff*(cutoff+1)//2
        rtn = CategoryCounter({'cphase': gates})

    return rtn


class CategoryCounter:

    def __init__(self, init_val=None):
        if init_val is None:
            self._d = defaultdict(lambda: 0)
        else:
            self._d = defaultdict(lambda: 0, init_val)

        self.ancillas = 0
        self.max_k = 0

    def __iadd__(self, val):
        for k, v in val.counts.items():
            self.counts[k] += v

        self.ancillas = max(val.ancillas, self.ancillas)
        self.max_k = max(self.max_k, val.max_k)
        return self

    def __rmul__(self, val):
        rtn = self.copy()
        for k in rtn.counts:
            rtn.counts[k] *= val
        return rtn

    def copy(self):
        rtn = type(self)(self._d)
        rtn.ancillas = self.ancillas
        rtn.max_k = self.max_k
        return rtn

    def include_k(self, k):
        if k > self.max_k:
            self.max_k = k

    def total_gates(self):
        return sum(self._d.values())

    def phase_rotations(self):
        return sum(self.counts[k] for k in self.counts if 'phase' in k)

    def count_gates(self, gate_list):
        return sum(self.counts[k] for k in self.counts if k in gate_list)

    @property
    def counts(self):
        return self._d


if __name__ == '__main__':
    main()
