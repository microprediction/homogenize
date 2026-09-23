# homogenize

Survival probability, equivalently a zero-coupon bond price, when the hazard rate is an
Ornstein–Uhlenbeck process whose mean level and volatility switch between two regimes:

    dx = κ(θ_s − x) dt + σ_s dW,    u(t, x, s) = E[ exp(−∫₀ᵗ x_r dr) | x₀ = x, s₀ = s ]

The package evaluates the fast-switching (homogenization) expansion to any order, including the
initial layer, for symmetric or unequal switching rates. It also solves the reduced two-state
linear ODE numerically, and simulates. Derivation and demonstrations:
[homogenization.microprediction.org](https://homogenization.microprediction.org/regime-switching.html).

## Install

    pip install homogenize

## Usage

```python
import math
from homogenize import RegimeSwitchingModel

model = RegimeSwitchingModel(kappa=2.0, thetas=[0.15, 0.02],
                             sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)], lmbd=5.0, order=6)
model.u(t=7.2, x=0.12, s=0)            # expansion through order 6
model.u(t=7.2, x=0.12, s=0, order=2)   # through order 2
model.numerical(t=7.2, x=0.12, s=0)    # numerical solution of the two-state ODE
model.series(t=7.2, x=0.12, s=0)       # contribution of each order
```

`lmbd` is the rate at which each regime switches to the other. For unequal occupancy pass
`rates=(rate 0 → 1, rate 1 → 0)` instead.

## How it works

Because κ does not switch, `u(t, x, s) = a_s(t) exp(−B(t) x)` with `B(t) = (1 − e^{−κt})/κ`, and `a`
solves a two-state linear ODE. With stationary probabilities `(p, q)`, total rate `γ`,
`s = p a₁ + q a₂` and `r = (a₁ − a₂)/s`:

    r' = D (1 + (q − p) r − p q r²) − γ r,   r(0) = 0,   D = G₁ − G₂,  G_i = −κ θ_i B + σ_i² B²/2
    log s = ∫₀ᵗ (p G₁ + q G₂ + p q D r)

Expanding `r` in `1/γ` gives terms that are polynomials in `E = 1 − e^{−κt}`, integrated in closed
form; an initial layer of width `1/γ` is solved to all orders in the span of `τᵏ e^{−jτ}`, `τ = γt`.
The error after `n` orders falls like `γ^{−(n+1)}`.

## Tests

    pip install -e .[test] && pytest
