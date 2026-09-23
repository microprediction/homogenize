import math
import pytest
from homogenize import RegimeSwitchingModel, simulate

BASE = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)])


@pytest.mark.parametrize("rates", [(20, 20), (14, 6)])
@pytest.mark.parametrize("s", [0, 1])
def test_orders_converge(rates, s):
    """ Error after n orders falls like gamma^-(n+1) as the total rate gamma doubles """
    t, x = 0.5, 0.12
    errs = []
    for scale in (1, 2):
        m = RegimeSwitchingModel(rates=(rates[0] * scale, rates[1] * scale), order=4, **BASE)
        ref = m.numerical(t, x, s, steps=20000)
        errs.append([m.u(t, x, s, order=n) - ref for n in range(4)])
    for n in range(4):
        assert abs(math.log2(abs(errs[0][n] / errs[1][n])) - (n + 1)) < 0.4


def test_expansion_matches_numerical():
    m = RegimeSwitchingModel(lmbd=5.0, order=6, **BASE)
    for t in (1.0, 4.0, 10.0):
        for s in (0, 1):
            assert abs(m.u(t, 0.12, s) - m.numerical(t, 0.12, s)) < 1e-7


def test_numerical_matches_simulation():
    m = RegimeSwitchingModel(rates=(2.0, 1.0), **BASE)
    est, se = simulate(2.0, 0.12, 0, rates=(2.0, 1.0), n_paths=200_000, n_steps=400, seed=3, **BASE)
    assert abs(est - m.numerical(2.0, 0.12, 0)) < 4 * se + 2e-4


def test_series_adds_up():
    m = RegimeSwitchingModel(lmbd=10.0, order=4, **BASE)
    assert abs(sum(m.series(3.0, 0.1, 1)) - m.u(3.0, 0.1, 1)) < 1e-15
