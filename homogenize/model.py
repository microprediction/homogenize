import math
import numpy as np
from homogenize.series import GeneralSeries


class RegimeSwitchingModel:
    """ Survival probability (equivalently a zero-coupon bond price) under a regime-switching
        Ornstein-Uhlenbeck hazard rate:

            dx = kappa (theta_s - x) dt + sigma_s dW,   s in {0, 1},
            u(t, x, s) = E[ exp(-int_0^t x_r dr) | x_0 = x, s_0 = s ].

        Switching rates: pass lmbd for a symmetric chain (each regime leaves at rate lmbd), or
        rates=(rate 0 -> 1, rate 1 -> 0) for unequal occupancy.

        u() evaluates the fast-switching expansion in 1/(total switching rate) through the given
        order, including the initial layer. numerical() solves the reduced two-state linear ODE,
        which needs no expansion and serves as a reference.
        See https://homogenization.microprediction.org/regime-switching.html
    """

    def __init__(self, kappa, thetas, sigmas, lmbd=None, rates=None, order=4):
        if rates is None:
            if lmbd is None:
                raise ValueError('Supply lmbd (symmetric switching) or rates=(r01, r10)')
            rates = (lmbd, lmbd)
        self.kappa, self.thetas, self.sigmas = kappa, list(thetas), list(sigmas)
        self.rates, self.order = tuple(rates), order
        self._series = GeneralSeries(kappa=kappa, thetas=self.thetas, sigmas=self.sigmas,
                                     lam12=rates[0], lam21=rates[1], order=max(order, 1))

    def tau(self, t):
        """ B(t) = (1 - exp(-kappa t)) / kappa """
        return (1 - math.exp(-self.kappa * t)) / self.kappa

    def u(self, t, x, s, order=None):
        """ Survival probability from the expansion, through the given order """
        order = self.order if order is None else order
        return self._series.u(t, x, s, order=order)

    def u0(self, t, x):
        """ Averaged model: Vasicek with occupancy-weighted mean level and variance """
        return self._series.u(t, x, 0, order=0)

    def series(self, t, x, s, order=None):
        """ Contribution of each order: [u0, u1 - u0, u2 - u1, ...] """
        order = self.order if order is None else order
        sums = [self._series.u(t, x, s, order=n) for n in range(order + 1)]
        return [sums[0]] + [b - a for a, b in zip(sums[:-1], sums[1:])]

    def numerical(self, t, x, s, steps=None):
        """ Solve a' = diag(G) a + Q a by fourth-order Runge-Kutta; u = a_s exp(-B x) """
        k, (r01, r10) = self.kappa, self.rates
        th = np.array(self.thetas, float)
        ss = np.array(self.sigmas, float) ** 2
        Q = np.array([[-r01, r01], [r10, -r10]], float)
        n = steps or max(400, int(math.ceil(t * max(40.0, 8 * (r01 + r10)))))
        h, a = t / n, np.ones(2)

        def f(r, v):
            b = (1 - math.exp(-k * r)) / k
            return (-k * th * b + 0.5 * ss * b * b) * v + Q @ v

        for i in range(n):
            r = i * h
            k1 = f(r, a)
            k2 = f(r + h / 2, a + h / 2 * k1)
            k3 = f(r + h / 2, a + h / 2 * k2)
            k4 = f(r + h, a + h * k3)
            a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return float(a[s] * math.exp(-self.tau(t) * x))
