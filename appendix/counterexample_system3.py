"""Appendix A: E[B(x+) | x] > B(x) at x = (0.79, -0.32) for System 3.

Interval bound propagation on 20 x 20 noise cells encloses E[B(x+)].
"""
import numpy as np
from mpmath import iv

from core.experiments import model

X = ('0.79', '-0.32')                                     # the counterexample state (exact decimals)
CELLS = 20                                                # n: the noise square is split into n x n cells


def load(prefix, weights):
    return [([[iv.mpf(float(w)) for w in row] for row in weights[f'{prefix}_net.{i}.weight']],
             [iv.mpf(float(v)) for v in weights[f'{prefix}_net.{i}.bias']]) for i in (0, 2, 4)]


def tanh(v):
    """Interval enclosure of tanh on the interval v (tanh is increasing; tanh(t) = 1 - 2 / (exp(2t) + 1))."""
    at = lambda t: 1 - 2 / (iv.exp(2 * iv.mpf(t)) + 1)                     # noqa: E731
    return iv.mpf([at(v.a).a, at(v.b).b])


def network(layers, z):
    """Interval enclosure of the network output for the interval inputs z (tanh hidden layers, linear output)."""
    for i, (W, b) in enumerate(layers):
        z = [sum((w * zj for w, zj in zip(row, z)), bias) for row, bias in zip(W, b)]
        if i < len(layers) - 1:
            z = [tanh(v) for v in z]
    return z[0]


def softplus(v):
    return iv.log(1 + iv.exp(v))


def inside_S(z1, z2):
    """Sound check that the whole box z1 x z2 lies in S (inside X and away from the three obstacles)."""
    in_X = z1.a >= -3.5 and z1.b <= 2 and z2.a >= -2 and z2.b <= 1
    off_boxes = z2.b < 0.1                                # both box obstacles have x2 >= 0.1
    off_disk = ((z1 + 1) ** 2 + (z2 + 1) ** 2).a > iv.mpf('0.16').b
    return in_X and off_boxes and off_disk


def main():
    weights = np.load(model('system3.npz'))
    B_net, pi_net = load('B', weights), load('pi', weights)
    x1, x2 = iv.mpf(X[0]), iv.mpf(X[1])
    u = 1.5 * tanh(network(pi_net, [x1, x2]))
    m1 = x1 + x2 / 10
    m2 = x2 + (x1 ** 3 / 3 - x1 - x2 + u) / 10
    Bx = softplus(network(B_net, [x1, x2]))
    lower = upper = iv.mpf(0)
    edges = [iv.mpf(-1) + iv.mpf(2) * i / CELLS for i in range(CELLS + 1)]
    for i in range(CELLS):
        for j in range(CELLS):
            xi1, xi2 = iv.mpf([edges[i].a, edges[i + 1].b]), iv.mpf([edges[j].a, edges[j + 1].b])
            z1, z2 = m1 + xi1 / 100, m2 + xi2 / 100
            assert inside_S(z1, z2), 'a successor box leaves S: refine or handle B = 1 outside S'
            B_box = softplus(network(B_net, [z1, z2]))
            lower += iv.mpf(B_box.a) / CELLS ** 2
            upper += iv.mpf(B_box.b) / CELLS ** 2
    print(f'x = ({X[0]}, {X[1]}),  u(x) in [{float(u.a):.6f}, {float(u.b):.6f}]')
    print(f'B(x)      in [{float(Bx.a):.9f}, {float(Bx.b):.9f}]')
    print(f'E[B(x+)]  in [{float(lower.a):.6f}, {float(upper.b):.6f}]   ({CELLS} x {CELLS} noise cells, all in S)')
    gap = lower.a - Bx.b
    print(f'lower bound of E[B(x+)] - upper bound of B(x) = {float(gap):.6f}  -> counterexample: {gap > 0}')


if __name__ == '__main__':
    main()
