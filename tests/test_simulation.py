from homogenize.fast import FastSimulator
from pprint import pprint

def test_simulate():
    rs = FastSimulator(kappa=6.0, thetas=[0.01, 0.5], sigmas=[0.01 * 0.01, 0.03 * 0.3], lmbd=2.0)
    moments = rs.moments(t=10,s=1,x=0.1,num=1000000)
    pprint(moments)

