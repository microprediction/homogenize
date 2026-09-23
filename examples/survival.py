import math
from homogenize import RegimeSwitchingModel

params = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)], lmbd=5.0)
model = RegimeSwitchingModel(order=6, **params)
t, x = 7.2, 0.12
for s in (0, 1):
    print(f"regime {s}: numerical {model.numerical(t, x, s):.10f}")
    for n in range(7):
        print(f"   order {n}: {model.u(t, x, s, order=n):.10f}")
