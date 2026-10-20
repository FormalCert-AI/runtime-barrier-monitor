"""Appendix A: E[B(x+) | x] > B(x) at x = (-0.07, 0.30) for System 2+input.

Exact rational calculation using the moments of xi ~ U[-0.1, 0.1] and b ~ U[0, 0.45].
"""
import json
import os
from fractions import Fraction

import sympy as sp

from core.config import MODELS

X = (Fraction(-7, 100), Fraction(3, 10))                  # the counterexample state


def barrier():
    with open(os.path.join(MODELS, 'system2_sos.json')) as f:
        cert = json.load(f, parse_float=Fraction)
    x1, x2 = sp.symbols('x1 x2')
    B = sum(sp.Rational(c.numerator, c.denominator) * x1 ** int(e[0]) * x2 ** int(e[1])
            for c, e in zip(cert['coef'], cert['expo']))
    return B, x1, x2


def expected_next(B, x1, x2, point):
    xi, b = sp.symbols('xi b')
    p1, p2 = (sp.Rational(v.numerator, v.denominator) for v in point)
    n1 = p1 - p2 / 50
    n2 = p2 + ((sp.Rational(4, 5) + xi) * p1 + 10 * (p1 ** 2 - sp.Rational(21, 100)) * p2 + b) / 100
    P = sp.Poly(sp.expand(B.subs({x1: n1, x2: n2}, simultaneous=True)), xi, b)
    m_xi = lambda n: sp.Rational(1, 10) ** n / (n + 1) if n % 2 == 0 else 0          # noqa: E731
    m_b = lambda n: sp.Rational(9, 20) ** n / (n + 1)                                  # noqa: E731
    E = sum(c * m_xi(i) * m_b(j) for (i, j), c in P.terms())
    return E, P, (n1, n2)


def main():
    B, x1, x2 = barrier()
    Bx = B.subs({x1: sp.Rational(X[0].numerator, X[0].denominator), x2: sp.Rational(X[1].numerator, X[1].denominator)})
    E, P, (n1, n2) = expected_next(B, x1, x2, X)
    # all successors stay in S (x+ is affine in (xi, b): check the corners)
    xi, b = sp.symbols('xi b')
    corners = [(s, t) for s in (sp.Rational(-1, 10), sp.Rational(1, 10)) for t in (0, sp.Rational(9, 20))]
    max_norm2 = max((n1 ** 2 + n2.subs({xi: s, b: t}) ** 2) for s, t in corners)
    print(f'x = ({X[0]}, {X[1]})')
    print(f'B(x)            = {sp.N(Bx, 12)}')
    print(f'E[B(x+)]        = {sp.N(E, 12)}')
    print(f'E[B(x+)] - B(x) = {sp.N(E - Bx, 12)}  (exact rational, > 0: {E - Bx > 0})')
    print(f'max |x+|^2 over the noise box = {sp.N(max_norm2, 6)} <= 1: {max_norm2 <= 1}')
    return B, Bx, E, P


if __name__ == '__main__':
    main()
