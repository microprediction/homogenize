# homogenize
Analytical formulas for survival in regime switching Orstein-Uhlenbeck processes

# Hi there
This package exists for one purpose only: estimation of survival probability for a triply stochastic regime switching process using a fairly advanced technique, asymptotic homogenization. I'm still tinkering
with it and there might yet be bugs ... so caveat emptor. The calculation is 
from the back of my phd thesis 18 years ago so I'm a bit rusty :) The technique is not like
a series expansion that is more accurate near t=0, rather it is more accurate for larger t. A little
counterintuitive. It needs to be used with care and I recommend running the simulations to get a sense. 

If you are truly interested there is a draft shared on Overleaf: https://www.overleaf.com/read/rqkgmnqfsvvm
If you discover a bug in the maple code for generating the series expansion or the
python code it is translated to I will be forever greatful. 

-Peter

# Usage 

To use the survival probability estimator

    from homogenize import RegimeSwitchingModel
    params = {'kappa': 2.0, 'thetas': [0.15, 0.02], 'sigmas': [math.sqrt(0.0555), math.sqrt(0.0055)], 'lmbd': 1.7}
    model = RegimeSwitchingModel(**params)
    model.u(x=0.12,t=7.2) 
    
To check against simulation

    from homogenize import demo
    demo() 
    
