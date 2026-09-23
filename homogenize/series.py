"""Fast-switching expansion, to any order, for a two-regime OU hazard with arbitrary switching rates.

dx = kappa (theta_i - x) dt + sigma_i dW, the regime switches 1 -> 2 at rate lam12 and 2 -> 1 at rate lam21.
gamma = lam12 + lam21, stationary probabilities p = lam21 / gamma (regime 1) and q = lam12 / gamma (regime 2).
u_i(t, x) = E[exp(-int_0^t x) | x_0 = x, y_0 = i] = exp(-B x) a_i(t), with B = (1 - exp(-kappa t)) / kappa and

    a' = diag(G_1, G_2) a + Q a,  G_i = -kappa theta_i B + sigma_i^2 B^2 / 2,  Q = gamma [[-q, q], [p, -p]].

In the coordinates s = p a_1 + q a_2 and r = (a_1 - a_2) / s the system becomes

    r' = D (1 + (q - p) r - p q r^2) - gamma r,  r(0) = 0,   D = G_1 - G_2,
    log s = int_0^t (p G_1 + q G_2 + p q D r),   a_1 = s (1 + q r),  a_2 = s (1 - p r).

With eps = 1 / gamma the outer series r = sum eps^n r_n has r_1 = D and
    r_{n+1} = (q - p) D r_n - p q D sum_{i+j=n} r_i r_j - r_n',
each r_n a polynomial in E = 1 - exp(-kappa t). The initial layer eta = r - r_outer, tau = gamma t, solves
    d eta / d tau = -eta + eps D(eps tau) [ (q - p) eta - p q (2 r_outer eta + eta^2) ],  eta(0) = -r_outer(0),
order by order in the span of tau^k exp(-j tau). For p = q = 1/2 and gamma = 2 lam this is all_orders.py.
"""
import math
from collections import defaultdict
import numpy as np
from numpy.polynomial import polynomial as P


def _taylor_of_poly_in_E(poly, kappa, K):
    """Taylor coefficients in r (degree <= K) of poly(E(r)), E = 1 - exp(-kappa r)."""
    e = np.zeros(K + 1)
    for i in range(1, K + 1):
        e[i] = -(-kappa) ** i / math.factorial(i)
    out = np.zeros(K + 1)
    power = np.zeros(K + 1)
    power[0] = 1.0
    for c in poly:
        out += c * power
        power = np.convolve(power, e)[:K + 1]
    return out


class ExpPoly:
    """sum c[(k, j)] tau^k exp(-j tau), j >= 1."""

    def __init__(self, terms=None):
        self.t = defaultdict(float)
        for key, c in (terms or {}).items():
            self.t[key] += c

    def __add__(self, o):
        r = ExpPoly(self.t)
        for key, c in o.t.items():
            r.t[key] += c
        return r

    def times_poly(self, coeffs):
        r = ExpPoly()
        for (k, j), c in self.t.items():
            for i, a in enumerate(coeffs):
                if a:
                    r.t[(k + i, j)] += c * a
        return r

    def __mul__(self, o):
        r = ExpPoly()
        for (k1, j1), c1 in self.t.items():
            for (k2, j2), c2 in o.t.items():
                r.t[(k1 + k2, j1 + j2)] += c1 * c2
        return r

    def value(self, tau):
        return sum(c * tau ** k * math.exp(-j * tau) for (k, j), c in self.t.items())

    def integral(self, T):
        """int_0^T"""
        tot = 0.0
        for (k, j), c in self.t.items():
            tail = math.exp(-j * T) * sum((j * T) ** i / math.factorial(i) for i in range(k + 1))
            tot += c * math.factorial(k) / j ** (k + 1) * (1 - tail)
        return tot

    @staticmethod
    def solve(f, y0):
        """y' + y = f, y(0) = y0."""
        y = ExpPoly()
        for (k, j), c in f.t.items():
            if j == 1:
                y.t[(k + 1, 1)] += c / (k + 1)
            else:
                b = j - 1  # p' - b p = tau^k
                for i in range(k + 1):
                    y.t[(k - i, j)] += -c * math.factorial(k) / math.factorial(k - i) / b ** (i + 1)
        y.t[(0, 1)] += y0 - y.value(0.0)
        return y


class GeneralSeries:
    def __init__(self, kappa, thetas, sigmas, lam12, lam21, order=6):
        k = self.k = kappa
        self.gamma = lam12 + lam21
        self.eps = 1 / self.gamma
        self.p = p = lam21 / self.gamma
        self.q = q = lam12 / self.gamma
        self.N = N = order

        def G(theta, ss):  # G(r) as a polynomial in E: -theta E + ss E^2 / (2 k^2)
            return np.array([0.0, -theta, ss / (2 * k * k)])

        G1 = G(thetas[0], sigmas[0] ** 2)
        G2 = G(thetas[1], sigmas[1] ** 2)
        self.D = D = G1 - G2
        self.Gbar = p * G1 + q * G2
        # outer
        r = [np.zeros(1), D.copy()]
        for n in range(1, N):
            conv = np.zeros(1)
            for i in range(1, n):
                conv = P.polyadd(conv, P.polymul(r[i], r[n - i]))
            nxt = P.polysub(P.polysub((q - p) * P.polymul(D, r[n]), p * q * P.polymul(D, conv)), self._ddt(r[n]))
            r.append(nxt)
        self.r = r
        self.log_outer = [None] + [p * q * P.polymul(D, r[n]) for n in range(1, N + 1)]
        # inner layer
        K = N + 2
        Dt = _taylor_of_poly_in_E(D, k, K)
        R = [None] + [_taylor_of_poly_in_E(r[n], k, K) for n in range(1, N + 1)]
        eta = [ExpPoly() for _ in range(N + 1)]
        for n in range(2, N + 1):
            f = ExpPoly()
            for a in range(1, n):
                if not Dt[a]:
                    continue
                m = n - 1 - a  # (q - p) eta term: eps^(1 + a + m)
                if m >= 2 and q != p:
                    f = f + eta[m].times_poly([0] * a + [(q - p) * Dt[a]])
                for b in range(1, n):
                    for c in range(0, n):
                        m = n - 1 - a - b - c  # -2 p q r_outer eta
                        if m >= 2 and R[b][c]:
                            f = f + eta[m].times_poly([0] * (a + c) + [-2 * p * q * Dt[a] * R[b][c]])
                for m1 in range(2, n):
                    m2 = n - 1 - a - m1  # -p q eta^2
                    if m2 >= 2:
                        f = f + (eta[m1] * eta[m2]).times_poly([0] * a + [-p * q * Dt[a]])
            eta[n] = ExpPoly.solve(f, -P.polyval(0.0, r[n]))
        self.eta, self.Dt = eta, Dt

    def _ddt(self, poly):
        return P.polymul(P.polyder(poly) if len(poly) > 1 else np.zeros(1), np.array([self.k, -self.k]))

    def _J(self, poly, t):
        k, tot = self.k, 0.0
        for n, c in enumerate(poly):
            if c:
                tot += c * (t + sum(math.comb(n, j) * (-1) ** j * (1 - math.exp(-j * k * t)) / (j * k) for j in range(1, n + 1)))
        return tot

    def u(self, t, x, regime, order=None):
        """Survival probability (bond price) through eps^order, starting in regime 0 or 1."""
        N = self.N if order is None else order
        e, E, T = self.eps, 1 - math.exp(-self.k * t), t / self.eps
        log_s = self._J(self.Gbar, t) + sum(e ** n * self._J(self.log_outer[n], t) for n in range(1, N + 1))
        for a in range(1, N + 1):  # layer: int_0^t p q D eta = eps int_0^T p q D(eps tau) eta dtau
            for m in range(2, N + 1):
                if 1 + a + m <= N and self.Dt[a]:
                    log_s += e ** (1 + a + m) * self.p * self.q * self.Dt[a] * self.eta[m].times_poly([0] * a + [1]).integral(T)
        rr = sum(e ** n * P.polyval(E, self.r[n]) for n in range(1, N + 1))
        rr += sum(e ** n * self.eta[n].value(T) for n in range(2, N + 1))
        factor = 1 + self.q * rr if regime == 0 else 1 - self.p * rr
        return math.exp(-E / self.k * x + log_s) * factor
