from homogenize.model import RegimeSwitchingModel
import numpy as np
import matplotlib.pyplot as plt
params = {'kappa': 3.0, 'thetas': [0.01, 0.05], 'sigmas': [0.005, 0.01], 'lmbd': 2.0}

approximator = RegimeSwitchingModel(**params)
ts = list(np.linspace(0.01,2,500))
plt.plot(ts,[approximator.v1_symmetric(t) for t in ts])
plt.plot(ts,[approximator.v1_antisymmetric(t) for t in ts])
plt.plot(ts,[approximator.v2_symmetric(t) for t in ts])
plt.plot(ts,[approximator.v2_antisymmetric(t) for t in ts])

plt.legend(['Symmetric 1st order','Antisymmetric 1st order','Symmetric 2nd order','Antisymmetric 2nd order'])
plt.xlabel('Time')
plt.ylabel('Multiplicative correction')
plt.show()