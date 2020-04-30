
import math
import numpy as np

class RegimeSwitchingModel():
    """ Analytic approximation to survival under regime switching OU process with stochastic mean level and volatility both
        dependent on the same two state regime s which switches with intensity lmbda. This class exists to compute an
        approximation to

          u(t,x,s) = E[ exp(-int_0^t x_s ds) | x(0)=x s(0)=s ]

        See https://www.overleaf.com/read/rqkgmnqfsvvm

    """

    def __init__(self,kappa,thetas,sigmas,lmbd):
        self.kappa, self.thetas, self.sigmas = kappa, thetas, sigmas
        self.lmbd = lmbd
        self.theta_bar   = 0.5*( thetas[0]+thetas[1] )
        self.ss_bar      = 0.5*( sigmas[0]**2+sigmas[1]**2 )
        self.theta_under = 0.5*( thetas[0]-thetas[1])
        self.ss_under    = 0.5*( sigmas[0]**2 -sigmas[1]**2 )
        self.epsilon     = 1/lmbd
        self.sigmas      = sigmas

    def as_latex(self):
        """ Describe tests parameters in latex suitable for plot titles """
        titles = ['$\kappa =$ ' + str(self.kappa),
                  '$\lambda=$ ' + str(self.lmbd),
                  '$\\theta_1=$ ' + str(self.thetas[0]),
                  '$\\theta_2=$ ' + str(self.thetas[1]),
                  '$\sigma_1=$ ' + str(self.sigmas[0]),
                  '$\sigma_2=$ ' + str(self.sigmas[1])]
        return r', '.join(titles)

    def tau(self,t):
        return (1-math.exp(-self.kappa*t))/self.kappa

    def a0(self,t):
        tau_ = self.tau(t)
        return (self.theta_bar - self.ss_bar/(2*self.kappa**2) )*(tau_-t) - (self.ss_bar*tau_**2)/(4*self.kappa)

    def series(self,t,x,s):
        u0 = self.u0(t,x)
        epsilon = 1 / self.lmbd
        pm = 1 if s == 0 else -1  # Plus or minus
        return [ u0,
                 u0*epsilon*self.v1_symmetric(t),
                 u0*epsilon*pm*self.v1_antisymmetric(t),
                 u0*epsilon**2*self.v2_symmetric(t),
                 u0*epsilon**2*pm*self.v2_antisymmetric(t) ]

    def u(self,t,x,s):
        """ Estimate survival probability within O(1/lmbd^3) """
        series = self.series(t=t,x=x,s=s)
        return np.nansum(series)

    def u0(self,t,x):
        return math.exp(self.a0(t)-self.tau(t)*x)

    def u0_check(self,t,x):
        K_     = self.kappa
        tau_   = (1 - np.exp(-K_ * t)) / K_
        ss_bar = self.ss_bar
        A = np.exp(  (self.theta_bar - ss_bar / (2 * K_ ** 2)) * (tau_ - t) - (ss_bar) / (4 * K_) * tau_ ** 2)
        return A * np.exp(-x * tau_)

    def v1_antisymmetric(self,t):
        """
         The following variable name replacements were made: b~ -> cg, k~ -> cg1, q~ -> cg4
        :param t:
        :return:
        """
        exp = math.exp
        cg = self.theta_under
        cg1 = self.kappa
        cg3 = self.ss_under
        return (cg * (0.1e1 - exp(-cg1 * t)) + cg3 * (0.1e1 - exp(-cg1 * t)) ** 2 / cg1 ** 2 / 0.2e1) ** 2

    def v1_symmetric(self,t):
        """
        Warning, the following variable name replacements were made: b~ -> cg, k~ -> cg1, q~ -> cg4
cg3 = ((-24 * cg ^ 2 * cg1 ^ 4 - 72 * cg * cg1 ^ 2 * cg4 - 36 * cg4 ^ 2) * exp(-(2 * cg1 * r)) + (96 * cg ^ 2 * cg1 ^ 4 + 144 * cg * cg1 ^ 2 * cg4 + 48 * cg4 ^ 2) * exp(-(cg1 * r)) + 0.16e2 * cg4 * (cg * cg1 ^ 2 + cg4) * exp(-(3 * cg1 * r)) + (48 * cg ^ 2 * cg1 ^ 5 * r) - (72 * cg ^ 2 * cg1 ^ 4) + (48 * cg * cg1 ^ 3 * cg4 * r) - (88 * cg * cg1 ^ 2 * cg4) + (12 * cg1 * cg4 ^ 2 * r) - 0.3e1 * (cg4 ^ 2) * exp(-(4 * cg1 * r)) - (25 * cg4 ^ 2)) / (cg1 ^ 5) / 0.48e2;
        :param t:
        :return:
        """
        cg  = self.theta_under
        cg1 = self.kappa
        cg4 = self.ss_under
        exp = math.exp
        return ((-24 * cg ** 2 * cg1 ** 4 - 72 * cg * cg1 ** 2 * cg4 - 36 * cg4 ** 2) * exp(-(2 * cg1 * t)) + (
                    96 * cg ** 2 * cg1 ** 4 + 144 * cg * cg1 ** 2 * cg4 + 48 * cg4 ** 2) * exp(
            -(cg1 * t)) + 0.16e2 * cg4 * (cg * cg1 ** 2 + cg4) * exp(-(3 * cg1 * t)) + (48 * cg ** 2 * cg1 ** 5 * t) - (
                           72 * cg ** 2 * cg1 ** 4) + (48 * cg * cg1 ** 3 * cg4 * t) - (88 * cg * cg1 ** 2 * cg4) + (
                           12 * cg1 * cg4 ** 2 * t) - 0.3e1 * (cg4 ** 2) * exp(-(4 * cg1 * t)) - (25 * cg4 ** 2)) / (
                          cg1**5) / 0.48e2

    def v2_antisymmetric(self,t):
        cg  = self.theta_under
        cg1 = self.kappa
        cg3 = self.ss_under
        exp = math.exp
        return -(-0.2e1 * (-24 * cg ** 2 * cg1 ** 4 - 72 * cg * cg1 ** 2 * cg3 - 36 * cg3 ** 2) * cg1 * exp(
            -(2 * cg1 * t)) - (96 * cg ** 2 * cg1 ** 4 + 144 * cg * cg1 ** 2 * cg3 + 48 * cg3 ** 2) * cg1 * exp(
            -(cg1 * t)) - 0.48e2 * cg3 * (cg * cg1 ** 2 + cg3) * cg1 * exp(-(3 * cg1 * t)) + (48 * cg ** 2 * cg1 ** 5) + (
                            48 * cg * cg1 ** 3 * cg3) + (12 * cg1 * cg3 ** 2) + 0.12e2 * (cg3 ** 2) * cg1 * exp(
            -(4 * cg1 * t))) / (cg1 ** 5) / 0.48e2 + (
                          cg * (0.1e1 - exp(-(cg1 * t))) + cg3 * (0.1e1 - exp(-(cg1 * t))) ** 2 / (cg1 ** 2) / 0.2e1) * (
                          (-24 * cg ** 2 * cg1 ** 4 - 72 * cg * cg1 ** 2 * cg3 - 36 * cg3 ** 2) * exp(-(2 * cg1 * t)) + (
                              96 * cg ** 2 * cg1 ** 4 + 144 * cg * cg1 ** 2 * cg3 + 48 * cg3 ** 2) * exp(
                      -(cg1 * t)) + 0.16e2 * cg3 * (cg * cg1 ** 2 + cg3) * exp(-(3 * cg1 * t)) + (
                                      48 * cg ** 2 * cg1 ** 5 * t) - (72 * cg ** 2 * cg1 ** 4) + (
                                      48 * cg * cg1 ** 3 * cg3 * t) - (88 * cg * cg1 ** 2 * cg3) + (
                                      12 * cg1 * cg3 ** 2 * t) - 0.3e1 * (cg3 ** 2) * exp(-(4 * cg1 * t)) - (
                                      25 * cg3 ** 2)) / (cg1 ** 5) / 0.48e2



    def v2_symmetric_approx(self,t):
        """
        :param t: Time  (maturity
        :return: Taylor series approx
        """
        cg = self.theta_under
        cg3 = self.ss_under
        cg1  = self.kappa
        return -cg1 ** 3 * cg ** 3 * t ** 4 / 0.4e1 + (
                    0.3e1 / 0.10e2 * cg1 ** 4 * cg ** 3 - 0.3e1 / 0.10e2 * cg1 ** 2 * cg ** 2 * cg3) * t ** 5 + (
                          cg1 ** 3 * cg ** 2 * cg3 / 0.2e1 + cg1 ** 4 * cg ** 4 / 0.18e2 - 0.5e1 / 0.24e2 * cg1 ** 5 * cg ** 3 - cg1 * cg * cg3 ** 2 / 0.8e1) * t ** 6 + (
                          0.3e1 / 0.28e2 * cg ** 3 * cg1 ** 6 - 0.13e2 / 0.28e2 * cg ** 2 * cg1 ** 4 * cg3 + 0.15e2 / 0.56e2 * cg * cg1 ** 2 * cg3 ** 2 - cg1 ** 5 * cg ** 4 / 0.12e2 - cg3 ** 3 / 0.56e2 + cg1 ** 3 * cg ** 3 * cg3 / 0.12e2) * t ** 7 + (
                          0.5e1 / 0.16e2 * cg ** 2 * cg1 ** 5 * cg3 + 0.101e3 / 0.1440e4 * cg1 ** 6 * cg ** 4 + 0.3e1 / 0.64e2 * cg1 * cg3 ** 3 - 0.43e2 / 0.960e3 * cg1 ** 7 * cg ** 3 - 0.5e1 / 0.16e2 * cg1 ** 3 * cg * cg3 ** 2 - 0.13e2 / 0.80e2 * cg1 ** 4 * cg ** 3 * cg3 + 0.23e2 / 0.480e3 * cg1 ** 2 * cg ** 2 * cg3 ** 2) * t ** 8 + (
                          -0.27e2 / 0.160e3 * cg1 ** 6 * cg ** 2 * cg3 + 0.25e2 / 0.144e3 * cg1 ** 5 * cg ** 3 * cg3 + cg1 * cg * cg3 ** 3 / 0.80e2 - 0.31e2 / 0.720e3 * cg1 ** 7 * cg ** 4 - 0.19e2 / 0.288e3 * cg1 ** 2 * cg3 ** 3 + 0.23e2 / 0.1440e4 * cg1 ** 8 * cg ** 3 + 0.25e2 / 0.96e2 * cg1 ** 4 * cg * cg3 ** 2 - 0.83e2 / 0.720e3 * cg1 ** 3 * cg ** 2 * cg3 ** 2) * t ** 9 + (
                          -0.373e3 / 0.2800e4 * cg ** 3 * cg1 ** 6 * cg3 + 0.7537e4 / 0.50400e5 * cg ** 2 * cg1 ** 4 * cg3 ** 2 - 0.43e2 / 0.1200e4 * cg * cg1 ** 2 * cg3 ** 3 - 0.121e3 / 0.24192e5 * cg ** 3 * cg1 ** 9 + 0.21e2 / 0.320e3 * cg1 ** 3 * cg3 ** 3 + 0.403e3 / 0.18900e5 * cg ** 4 * cg1 ** 8 + cg3 ** 4 / 0.800e3 + 0.37e2 / 0.480e3 * cg ** 2 * cg1 ** 7 * cg3 - 0.331e3 / 0.1920e4 * cg * cg1 ** 5 * cg3 ** 2) * t ** 10 + (
                          0.23e2 / 0.420e3 * cg1 ** 3 * cg * cg3 ** 3 - 0.181e3 / 0.20160e5 * cg1 ** 9 * cg ** 4 - 0.463e3 / 0.3360e4 * cg1 ** 5 * cg ** 2 * cg3 ** 2 - 0.6821e4 / 0.221760e6 * cg1 ** 8 * cg ** 2 * cg3 + 0.47e2 / 0.576e3 * cg1 ** 7 * cg ** 3 * cg3 + 0.311e3 / 0.221760e6 * cg1 ** 10 * cg ** 3 - cg1 * cg3 ** 4 / 0.240e3 - 0.1087e4 / 0.21120e5 * cg1 ** 4 * cg3 ** 3 + 0.135e3 / 0.1408e4 * cg1 ** 6 * cg * cg3 ** 2) * t ** 11

    def v2_symmetric(self,t):
        """ Second order symmetric correction term """
        cg = self.theta_under
        cg1 = self.kappa
        cg3 = self.ss_under
        exp = math.exp
        # terms = [ 0.25e2 / 0.8e1 / cg1 ** 3 * cg ** 2 * cg3  ,    0.137e3 / 0.80e2 / cg1 ** 5 * cg * cg3 ** 2 , 0.9e1 / 0.2e1 / cg1 ** 3 / exp(cg1 * t) ** 2 * cg ** 2 * cg3 - 0.6e1 / cg1 ** 3 / exp(cg1 * t) * cg ** 2 * cg3 - 0.2e1 / cg1 ** 3 * cg3 / exp(cg1 * t) ** 3 * cg ** 2 - 0.3e1 / 0.2e1 / cg1 ** 2 * cg ** 2 * cg3 * t - 0.3e1 / 0.4e1 / cg1 ** 4 * cg * cg3 ** 2 * t , 0.3e1 / 0.8e1 / cg1 ** 3 * cg ** 2 / exp(cg1 * t) ** 4 * cg3 , 0.15e2 / 0.4e1 / cg1 ** 5 * cg / exp(cg1 * t) ** 2 * cg3 ** 2 - 0.15e2 / 0.4e1 / cg1 ** 5 * cg / exp(cg1 * t) * cg3 ** 2 - 0.5e1 / 0.2e1 / cg1 ** 5 * cg * cg3 ** 2 / exp(cg1 * t) ** 3 , 0.15e2 / 0.16e2 / cg1 ** 5 * cg * cg3 ** 2 / exp(cg1 * t) ** 4 - 0.3e1 / 0.20e2 / cg1 ** 5 * cg / exp(cg1 * t) ** 5 * cg3 ** 2 - 0.47e2 / 0.48e2 / cg1 ** 7 * cg * cg3 ** 3 * t - 0.131e3 / 0.48e2 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t - 0.10e2 / 0.3e1 / cg1 ** 3 * cg ** 3 * cg3 * t , 0.59e2 / 0.288e3 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 6 - 0.15e2 / 0.16e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 5 , 0.165e3 / 0.32e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 2 - 0.163e3 / 0.48e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) - 0.1e1 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 7 / 0.48e2 , 0.247e3 / 0.96e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 4 - 0.653e3 / 0.144e3 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 3 , 0.1e1 / cg1 ** 6 * cg * cg3 ** 3 * t ** 2 / 0.4e1 , 0.1021e4 / 0.96e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 2 - 0.193e3 / 0.24e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) , 0.25e2 / 0.288e3 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 6 - 0.19e2 / 0.24e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 5 , 0.313e3 / 0.96e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 4 - 0.137e3 / 0.18e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 3 , 0.3e1 / 0.4e1 / cg1 ** 4 * cg ** 2 * cg3 ** 2 * t ** 2 - 0.49e2 / 0.6e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) - 0.1e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 5 / 0.6e1 , 0.17e2 / 0.12e2 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 4 - 0.5e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 3 , 0.55e2 / 0.6e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 2 , 0.1e1 / cg1 ** 2 * cg ** 3 * cg3 * t ** 2 - 0.19e2 / 0.8e1 / exp(cg1 * t) ** 2 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t , 0.7e1 / 0.4e1 / exp(cg1 * t) / cg1 ** 7 * cg * cg3 ** 3 * t , 0.2e1 / 0.3e1 / exp(cg1 * t) ** 3 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t , 0.1e1 / exp(cg1 * t) ** 3 / cg1 ** 3 * cg ** 3 * cg3 * t / 0.3e1 , 0.9e1 / 0.2e1 / exp(cg1 * t) / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t - 0.2e1 / exp(cg1 * t) ** 2 / cg1 ** 3 * cg ** 3 * cg3 * t , cg ** 4 * t ** 2 / 0.2e1 - cg ** 3 * t , 0.3e1 / 0.2e1 / cg1 / exp(cg1 * t) ** 2 * cg ** 3 - 0.3e1 / cg1 / exp(cg1 * t) * cg ** 3 - 0.1e1 / cg1 * cg ** 3 / exp(cg1 * t) ** 3 / 0.3e1 , 0.15e2 / 0.16e2 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 2 - 0.3e1 / 0.4e1 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) - 0.5e1 / 0.6e1 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 3 - 0.1e1 / cg1 ** 6 * cg3 ** 3 * t / 0.8e1 , 0.15e2 / 0.32e2 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 4 - 0.3e1 / 0.20e2 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 5 , 0.1e1 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 6 / 0.48e2 - 0.25e2 / 0.192e3 / cg1 ** 9 * cg3 ** 4 * t - 0.3e1 / 0.2e1 / cg1 * cg ** 4 * t , 0.1e1 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 8 / 0.512e3 - 0.1e1 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 7 / 0.48e2 , 0.59e2 / 0.576e3 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 6 - 0.5e1 / 0.16e2 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 5 , 0.497e3 / 0.768e3 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 4 - 0.133e3 / 0.144e3 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 3 , 0.57e2 / 0.64e2 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 2 - 0.25e2 / 0.48e2 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) , 0.1e1 / cg1 ** 8 * cg3 ** 4 * t ** 2 / 0.32e2 , 0.1e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) ** 4 / 0.8e1 - 0.1e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) ** 3 , 0.11e2 / 0.4e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) ** 2 - 0.3e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) , 0.1e1 / exp(cg1 * t) / cg1 ** 9 * cg3 ** 4 * t / 0.4e1 , 0.275e3 / 0.288e3 / cg1 ** 8 * cg * cg3 ** 3 , 0.709e3 / 0.288e3 / cg1 ** 6 * cg ** 2 * cg3 ** 2 , 0.2e1 / exp(cg1 * t) / cg1 * cg ** 4 * t - 0.1e1 / exp(cg1 * t) ** 2 / cg1 * cg ** 4 * t / 0.2e1 , 0.11e2 / 0.4e1 / cg1 ** 4 * cg ** 3 * cg3 - 0.1e1 / exp(cg1 * t) ** 4 / cg1 ** 9 * cg3 ** 4 * t / 0.64e2 , 0.1e1 / exp(cg1 * t) ** 3 / cg1 ** 9 * cg3 ** 4 * t / 0.12e2 - 0.3e1 / 0.16e2 / exp(cg1 * t) ** 2 / cg1 ** 9 * cg3 ** 4 * t , 0.11e2 / 0.6e1 / cg1 * cg ** 3 , 0.49e2 / 0.160e3 / cg1 ** 7 * cg3 ** 3 , 0.5e1 / exp(cg1 * t) / cg1 ** 3 * cg ** 3 * cg3 * t - 0.1e1 / exp(cg1 * t) ** 4 / cg1 ** 7 * cg * cg3 ** 3 * t / 0.16e2 , 0.5e1 / 0.12e2 / exp(cg1 * t) ** 3 / cg1 ** 7 * cg * cg3 ** 3 * t - 0.1e1 / exp(cg1 * t) ** 4 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t / 0.16e2 - 0.9e1 / 0.8e1 / exp(cg1 * t) ** 2 / cg1 ** 7 * cg * cg3 ** 3 * t , 0.9e1 / 0.8e1 / cg1 ** 2 * cg ** 4 , 0.625e3 / 0.4608e4 / cg1 ** 10 * cg3 ** 4 ]
        terms = [0.25e2 / 0.8e1 / cg1 ** 3 * cg ** 2 * cg3, 0.137e3 / 0.80e2 / cg1 ** 5 * cg * cg3 ** 2,
                 0.9e1 / 0.2e1 / cg1 ** 3 / exp(cg1 * t) ** 2 * cg ** 2 * cg3 - 0.6e1 / cg1 ** 3 / exp(
                     cg1 * t) * cg ** 2 * cg3 - 0.2e1 / cg1 ** 3 * cg3 / exp(
                     cg1 * t) ** 3 * cg ** 2 - 0.3e1 / 0.2e1 / cg1 ** 2 * cg ** 2 * cg3 * t - 0.3e1 / 0.4e1 / cg1 ** 4 * cg * cg3 ** 2 * t,
                 0.3e1 / 0.8e1 / cg1 ** 3 * cg ** 2 / exp(cg1 * t) ** 4 * cg3,
                 0.15e2 / 0.4e1 / cg1 ** 5 * cg / exp(cg1 * t) ** 2 * cg3 ** 2 - 0.15e2 / 0.4e1 / cg1 ** 5 * cg / exp(
                     cg1 * t) * cg3 ** 2 - 0.5e1 / 0.2e1 / cg1 ** 5 * cg * cg3 ** 2 / exp(cg1 * t) ** 3,
                 0.15e2 / 0.16e2 / cg1 ** 5 * cg * cg3 ** 2 / exp(cg1 * t) ** 4 - 0.3e1 / 0.20e2 / cg1 ** 5 * cg / exp(
                     cg1 * t) ** 5 * cg3 ** 2 - 0.47e2 / 0.48e2 / cg1 ** 7 * cg * cg3 ** 3 * t - 0.131e3 / 0.48e2 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t - 0.10e2 / 0.3e1 / cg1 ** 3 * cg ** 3 * cg3 * t,
                 0.59e2 / 0.288e3 / cg1 ** 8 * cg * cg3 ** 3 / exp(
                     cg1 * t) ** 6 - 0.15e2 / 0.16e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 5,
                 0.165e3 / 0.32e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(
                     cg1 * t) ** 2 - 0.163e3 / 0.48e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(
                     cg1 * t) - 0.1e1 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 7 / 0.48e2,
                 0.247e3 / 0.96e2 / cg1 ** 8 * cg * cg3 ** 3 / exp(
                     cg1 * t) ** 4 - 0.653e3 / 0.144e3 / cg1 ** 8 * cg * cg3 ** 3 / exp(cg1 * t) ** 3,
                 0.1e1 / cg1 ** 6 * cg * cg3 ** 3 * t ** 2 / 0.4e1,
                 0.1021e4 / 0.96e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(
                     cg1 * t) ** 2 - 0.193e3 / 0.24e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t),
                 0.25e2 / 0.288e3 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(
                     cg1 * t) ** 6 - 0.19e2 / 0.24e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 5,
                 0.313e3 / 0.96e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(
                     cg1 * t) ** 4 - 0.137e3 / 0.18e2 / cg1 ** 6 * cg ** 2 * cg3 ** 2 / exp(cg1 * t) ** 3,
                 0.3e1 / 0.4e1 / cg1 ** 4 * cg ** 2 * cg3 ** 2 * t ** 2 - 0.49e2 / 0.6e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(
                     cg1 * t) - 0.1e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 5 / 0.6e1,
                 0.17e2 / 0.12e2 / cg1 ** 4 * cg ** 3 * cg3 / exp(
                     cg1 * t) ** 4 - 0.5e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 3,
                 0.55e2 / 0.6e1 / cg1 ** 4 * cg ** 3 * cg3 / exp(cg1 * t) ** 2,
                 0.1e1 / cg1 ** 2 * cg ** 3 * cg3 * t ** 2 - 0.19e2 / 0.8e1 / exp(
                     cg1 * t) ** 2 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t,
                 0.7e1 / 0.4e1 / exp(cg1 * t) / cg1 ** 7 * cg * cg3 ** 3 * t,
                 0.2e1 / 0.3e1 / exp(cg1 * t) ** 3 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t,
                 0.1e1 / exp(cg1 * t) ** 3 / cg1 ** 3 * cg ** 3 * cg3 * t / 0.3e1,
                 0.9e1 / 0.2e1 / exp(cg1 * t) / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t - 0.2e1 / exp(
                     cg1 * t) ** 2 / cg1 ** 3 * cg ** 3 * cg3 * t, cg ** 4 * t ** 2 / 0.2e1 - cg ** 3 * t,
                 0.3e1 / 0.2e1 / cg1 / exp(cg1 * t) ** 2 * cg ** 3 - 0.3e1 / cg1 / exp(
                     cg1 * t) * cg ** 3 - 0.1e1 / cg1 * cg ** 3 / exp(cg1 * t) ** 3 / 0.3e1,
                 0.15e2 / 0.16e2 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 2 - 0.3e1 / 0.4e1 / cg1 ** 7 * cg3 ** 3 / exp(
                     cg1 * t) - 0.5e1 / 0.6e1 / cg1 ** 7 * cg3 ** 3 / exp(
                     cg1 * t) ** 3 - 0.1e1 / cg1 ** 6 * cg3 ** 3 * t / 0.8e1,
                 0.15e2 / 0.32e2 / cg1 ** 7 * cg3 ** 3 / exp(cg1 * t) ** 4 - 0.3e1 / 0.20e2 / cg1 ** 7 * cg3 ** 3 / exp(
                     cg1 * t) ** 5, 0.1e1 / cg1 ** 7 * cg3 ** 3 / exp(
                cg1 * t) ** 6 / 0.48e2 - 0.25e2 / 0.192e3 / cg1 ** 9 * cg3 ** 4 * t - 0.3e1 / 0.2e1 / cg1 * cg ** 4 * t]
        try:
            difficult_terms = [0.1e1 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 8 / 0.512e3 - 0.1e1 / cg1 ** 10 * cg3 ** 4 / exp(
                     cg1 * t) ** 7 / 0.48e2, 0.59e2 / 0.576e3 / cg1 ** 10 * cg3 ** 4 / exp(
                cg1 * t) ** 6 - 0.5e1 / 0.16e2 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 5,
                 0.497e3 / 0.768e3 / cg1 ** 10 * cg3 ** 4 / exp(
                     cg1 * t) ** 4 - 0.133e3 / 0.144e3 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t) ** 3,
                 0.57e2 / 0.64e2 / cg1 ** 10 * cg3 ** 4 / exp(
                     cg1 * t) ** 2 - 0.25e2 / 0.48e2 / cg1 ** 10 * cg3 ** 4 / exp(cg1 * t),
                 0.1e1 / cg1 ** 8 * cg3 ** 4 * t ** 2 / 0.32e2,
                 0.1e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) ** 4 / 0.8e1 - 0.1e1 / cg1 ** 2 * cg ** 4 / exp(
                     cg1 * t) ** 3,
                 0.11e2 / 0.4e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t) ** 2 - 0.3e1 / cg1 ** 2 * cg ** 4 / exp(cg1 * t),
                 0.1e1 / exp(cg1 * t) / cg1 ** 9 * cg3 ** 4 * t / 0.4e1, 0.275e3 / 0.288e3 / cg1 ** 8 * cg * cg3 ** 3,
                 0.709e3 / 0.288e3 / cg1 ** 6 * cg ** 2 * cg3 ** 2,
                 0.2e1 / exp(cg1 * t) / cg1 * cg ** 4 * t - 0.1e1 / exp(cg1 * t) ** 2 / cg1 * cg ** 4 * t / 0.2e1,
                 0.11e2 / 0.4e1 / cg1 ** 4 * cg ** 3 * cg3 - 0.1e1 / exp(
                     cg1 * t) ** 4 / cg1 ** 9 * cg3 ** 4 * t / 0.64e2,
                 0.1e1 / exp(cg1 * t) ** 3 / cg1 ** 9 * cg3 ** 4 * t / 0.12e2 - 0.3e1 / 0.16e2 / exp(
                     cg1 * t) ** 2 / cg1 ** 9 * cg3 ** 4 * t, 0.11e2 / 0.6e1 / cg1 * cg ** 3,
                 0.49e2 / 0.160e3 / cg1 ** 7 * cg3 ** 3,
                 0.5e1 / exp(cg1 * t) / cg1 ** 3 * cg ** 3 * cg3 * t - 0.1e1 / exp(
                     cg1 * t) ** 4 / cg1 ** 7 * cg * cg3 ** 3 * t / 0.16e2,
                 0.5e1 / 0.12e2 / exp(cg1 * t) ** 3 / cg1 ** 7 * cg * cg3 ** 3 * t,
                 - 0.1e1 / exp(
                     cg1 * t) ** 4 / cg1 ** 5 * cg ** 2 * cg3 ** 2 * t / 0.16e2 - 0.9e1 / 0.8e1 / exp(
                     cg1 * t) ** 2 / cg1 ** 7 * cg * cg3 ** 3 * t, 0.9e1 / 0.8e1 / cg1 ** 2 * cg ** 4 ]
        except:
            print('Warning ... discarding some terms')
            difficult_terms = [0]
        return sum(terms+difficult_terms)







