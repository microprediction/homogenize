from homogenize.model import RegimeSwitchingModel
from homogenize.slow import SlowSimulator
import math
import matplotlib.pyplot as plt

def run():
    params = {'kappa':6.0,'thetas':[0.01,0.1],'sigmas':[0.005,0.05],'lmbd':2.0}
    x,s,t = 0.015, 1, 10.0
    model = SlowSimulator(x=x, t=t, s=s, **params)
    u     = model.u(t=t,x=x,s=s)
    u0    = model.u0(t=t,x=x)
    approximations = list()
    vasicek = list()
    for _ in range(10):
        model.simulate(num=10000)
        model.price()
        approximations.append(model.u_estimte)
        plt.plot(approximations,'o')
        plt.plot([u for _ in approximations] )
        plt.plot([u0 for _ in approximations])
        plt.show(block=False)
        plt.pause(0.2)


if __name__=="__main__":
    run()
