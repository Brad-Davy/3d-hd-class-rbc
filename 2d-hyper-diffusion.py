"""
Dedalus script solving the 2d diffusion equation. The script uses Fourier basis 
functions in the x axis and Fourier basis functions in the y axis. It implements 
both standard diffusion and hyperdiffusion from the 'dedalusHyperDiffusion' class. T
he amplitudes of the coeficients of the different diffusion methods are then compared.

    dt(v) = dx(dx(v)) + dy(dy(v))
    dt(u) = H(u) = ν (dx(dx(u)) + dy(dy(u)))
    
    Where,
        
        ν = q^(k - k_0) if k > k_0
        k = (k_x^2 + k_y^2)^0.5
                
        ν = 1 else
Usage:
    2d-hyper-diffusion.py [--oneOver=<oneOver>]
    2d-hyper-diffusion.py -h | --help

Options:
    --h --help               Display this help message
    --oneOver=<oneOver>      Plotting type [default: 0]
"""

from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt
from dedalusHyperDiffusion import hyperDiffusion
from colours import *
from docopt import docopt


# =============================================================================
# Deal with docopt arguments.
# =============================================================================

args = docopt(__doc__)
oneOverOther = int(args['--oneOver'])
if oneOverOther == 1:
    oneOverOther = True
    
else:
    oneOverOther = False
    
# =============================================================================
# Define the basis, domain and the problem.
# =============================================================================

xbasis = de.Fourier('x',128, interval=(-np.pi,np.pi), dealias=1)
ybasis = de.Fourier('y',128, interval=(-np.pi,np.pi), dealias=1)
domain = de.Domain([xbasis, ybasis],np.float64)
problem = de.IVP(domain, variables=['u', 'v', 'dif','hypdif'])

# =============================================================================
# Define the hyperdiffusion operator
# =============================================================================

de.operators.parseables['hyperDiffusion'] = hyperDiffusion
problem.substitutions['H'] = "hyperDiffusion"

# =============================================================================
# Define the equations and relevant subsitutions to access later for plotting.
# =============================================================================

problem.add_equation("dt(u) + H(u) = 0")
problem.add_equation("dt(v) - dx(dx(v)) - dy(dy(v))= 0")
problem.add_equation("dif = dx(dx(v)) + dy(dy(v))")
problem.add_equation("hypdif = -H(u)")

# =============================================================================
# Define the solver, and create access to variables.
# =============================================================================

solver = problem.build_solver(de.timesteppers.SBDF2)
x = domain.grid(0)
u = solver.state['u']
v = solver.state['v']
dif = solver.state['dif']
hypdif = solver.state['hypdif']

# =============================================================================
# Construct IC and set solver parameters
# =============================================================================


u['c'] = np.ones(np.shape(u['c']), dtype = float)
v['c'] = np.ones(np.shape(v['c']), dtype = float)

dt = 1e-3
solver.stop_iteration = 2

# =============================================================================
# Main loop, not nessercary for plotting.
# =============================================================================

while solver.ok:
    # take a time step
    solver.step(dt)
    # every so many steps, save results for plotting.
    if solver.iteration % 2 ==0:
        u.set_scales(1)

# =============================================================================
# Deals with plotting.        
# =============================================================================

fig = plt.figure(figsize=(10,5))
plt.xlabel('$k_x$')
plt.ylabel('Amplitude')

if oneOverOther:
    dif['c'].real[0] = hypdif['c'].real[0] = 1
    sumOver = np.sum((hypdif['c'].real/dif['c'].real), axis = 0)
    plt.plot(sumOver, label = '$ \\frac{\\nu_0q^{i - i_0} \\nabla^2}{\\nabla^2} $', lw = 2, color = CB91_Blue)  

else:
    sumOverDif = np.sum(dif['c'].real, axis = 1)
    sumOverHypDif = np.sum(hypdif['c'].real, axis = 1)
    plt.plot(sumOverDif, label = '$ \\nabla^2 $', lw = 2, color = CB91_Blue)  
    plt.plot(sumOverHypDif, label = '$ \\nu_0q^{i - i_0} \\nabla^2$', lw = 2, color = CB91_Green)  


plt.legend()
plt.show()



