from homogenize.model import RegimeSwitchingModel
import numpy as np
import matplotlib.pyplot as plt
params = {'kappa': 4.0, 'thetas': [0.02, 0.2], 'sigmas': [0.015, 0.1], 'lmbd': 2.0}

approximator = RegimeSwitchingModel(**params)
ts = list(np.linspace(0.01,5,500))
SCALE = 10000
plt.plot(ts,[SCALE*approximator.v1_symmetric(t) for t in ts])
plt.plot(ts,[SCALE*approximator.v1_antisymmetric(t) for t in ts])
plt.plot(ts,[SCALE*(1/approximator.lmbd)*approximator.v2_symmetric(t) for t in ts])
plt.plot(ts,[SCALE*(1/approximator.lmbd)*approximator.v2_antisymmetric(t) for t in ts])

plt.legend(['Symmetric 1st order','Antisymmetric 1st order','Symmetric 2nd order','Antisymmetric 2nd order'])
plt.xlabel('Time')
plt.ylabel('Multiplicative correction (bps)')
plt.grid()
plt.title(approximator.as_latex())

plt.show()