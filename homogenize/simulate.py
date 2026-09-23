import math
import numpy as np


def simulate(t, x, s, kappa, thetas, sigmas, rates, n_paths=100_000, n_steps=400, seed=0):
    """ Monte Carlo survival probability: exact-in-law OU steps, regime switches checked each step,
        trapezoid rule for the integral. Returns (estimate, standard error). """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    th, sg = np.asarray(thetas, float), np.asarray(sigmas, float)
    leave = 1 - np.exp(-np.asarray(rates, float) * dt)
    xs = np.full(n_paths, float(x))
    ys = np.full(n_paths, s, dtype=int)
    integral = np.zeros(n_paths)
    decay = math.exp(-kappa * dt)
    sd = math.sqrt((1 - decay ** 2) / (2 * kappa))
    for _ in range(n_steps):
        prev = xs
        xs = th[ys] + (xs - th[ys]) * decay + sg[ys] * sd * rng.standard_normal(n_paths)
        integral += 0.5 * dt * (prev + xs)
        ys = np.where(rng.random(n_paths) < leave[ys], 1 - ys, ys)
    v = np.exp(-integral)
    return float(v.mean()), float(v.std() / math.sqrt(n_paths))
