import numpy as np
import random, math
from homogenize.model import RegimeSwitchingModel
from homogenize.running import RunningStats
import matplotlib.pyplot as plt



class SlowSimulator(RegimeSwitchingModel):

    # Class for checking numerical against analytic

    def __init__(self,kappa,thetas,sigmas,lmbd, ts,x,s):
        super().__init__(kappa=kappa, thetas=thetas,sigmas=sigmas,lmbd=lmbd)
        assert ts[0]>0
        self.ts        = ts
        self.x         = x
        self.s         = s
        self.integrals = [ RunningStats() for t in ts]
        self.exp_int   = [ RunningStats() for t in ts]
        self.survival  = [ 1 for _ in ts ]

    def simulate(self,num):
        dts = np.diff([0] + self.ts)
        for _ in range(num):
            intx = 0
            xt = self.x
            st = self.s
            for k, t,dt in zip(range(len(self.ts)), self.ts, dts):
                 xt_prev = xt
                 xt   = xt + self.kappa*(self.thetas[st]-xt)*dt + self.sigmas[st]*np.random.randn()*math.sqrt(dt)
                 intx = intx + dt*(xt_prev+xt)/2
                 self.integrals[k].push(intx)
                 self.exp_int[k].push(math.exp(-intx))
                 if np.random.rand()<dt*self.lmbd:
                     st = 1-st
        self.survival  = [ math.exp(-rs.mean() + 0.5 * rs.variance()) for rs in self.integrals]
        self.survival1 = [ rs.mean() for rs in self.exp_int ]
        self.survival_std = [ math.sqrt( rs.variance() ) for rs in self.exp_int ]

    def check(self, batch_size):
        """ Simulate a plot """
        dumb   = [ math.exp(-self.theta_bar*t) for t in self.ts ]
        us     = [ self.u(t,self.x,self.s) for t in self.ts ]
        u0s    = [ self.u0(t, self.x) for t in self.ts]
        u0_checks = [self.u0_check(t, self.x) for t in self.ts]
        series = [ self.series(t,self.x,self.s) for t in self.ts ]
        consecutive = list()
        for k in range(6):
            consecutive.append( [ sum(sr[:k+1]) for sr in series ] )

        fig, axs = plt.subplots(nrows=2,ncols=2)
        fig.suptitle( self.as_latex() )
        for batch_no in range(10000):
            axs[0][0].clear()
            self.simulate(num=batch_size)
            axs[0][0].plot(self.ts, dumb, self.ts, self.survival, self.ts, u0_checks, self.ts, consecutive[0], self.ts, consecutive[1], self.ts,consecutive[2], self.ts, consecutive[3], self.ts, consecutive[4] )
            axs[0][0].legend(['Dumb','Simulation','Check 0','Approx 0','Approx 1','Approx 2','Approx 3','Approx 4'])
            axs[0][0].set_xlabel('After '+str(batch_no*batch_size)+' paths.')
            axs[0][0].grid()
            axs[0][0].figure

            axs[0][1].clear()
            axs[0][1].plot(self.ts, [ 10000 * (s1 - s2) for s1, s2 in zip( u0s,  self.survival)])
            how_many = 3  # 1..4
            for k in range(1,how_many+1): #
                axs[0][1].plot(self.ts, [ 10000 * (s1 - s2) for s1, s2 in zip( consecutive[k], self.survival)])
            axs[0][1].set_xlabel('Discrepancy versus simulation')
            axs[0][1].legend(['u0 - sim']+['Approx '+str(k)+' - sim' for k in range(1,5)][:how_many] )
            axs[0][1].set_ylabel('bps')
            axs[0][1].plot(self.ts, [ 10000*2*sd/math.sqrt(batch_size*(batch_no+1)) for sd in self.survival_std], 'k-')
            axs[0][1].plot(self.ts, [-10000*2*sd/math.sqrt(batch_size*(batch_no+1)) for sd in self.survival_std], 'k-')
            axs[0][1].set_title('Two standard deviation envelope')
            axs[0][1].grid()
            axs[0][1].figure

            axs[1][0].clear()
            axs[1][0].plot(self.ts, [10000 * (u1 - u2) for u1, u2 in zip(u0s, u0_checks)])
            axs[1][0].plot(self.ts, [10000 * (u1 - u2) for u1, u2 in zip(u0s, dumb)])
            axs[1][0].plot(self.ts, [10000 * (u1 - u2) for u1, u2 in zip(us, dumb)])
            axs[1][0].grid()
            axs[1][0].set_xlabel('Adjustments')
            axs[1][0].legend(['u0 - u0_check','u0 - dumb','u - dumb'])
            axs[1][0].set_ylabel('bps')
            axs[1][0].figure


            plt.show(block=False)
            plt.pause(1)



def run():
    params = {'kappa': 2.0, 'thetas': [0.15, 0.02], 'sigmas': [math.sqrt(0.0555), math.sqrt(0.0055)], 'lmbd': 1.7}
    x, s = 0.11, 1
    ts = [0.01 * (k + 1) for k in range(2500)]
    model = SlowSimulator(x=x, ts=ts, s=s, **params)
    model.check(batch_size=1000)


if __name__=="__main__":
    run()





