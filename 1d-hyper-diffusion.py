"""
Dedalus script solving the 1d diffusion equation. The script uses Fourier basis 
functions in the x axis. It implements both standard diffusion and hyperdiffusion 
from the 'dedalusHyperDiffusion' class. The amplitudes of the coeficients of the 
different diffusion methods are then compared.

    dt(v) = dx(dx(v))
    dt(u) = H(u) = ν dx(dx(u))
    
    Where,
        
        ν = q^(i - i_0) if i > i_0
        
        ν = 1 else
Usage:
    1d-hyper-diffusion.py [--oneOver=<oneOver>]
    1d-hyper-diffusion.py -h | --help

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
domain = de.Domain([xbasis],np.float64)
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
problem.add_equation("dt(v) - dx(dx(v)) = 0")
problem.add_equation("dif = dx(dx(v))")
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

u['c'] = np.ones(int(len(u['g'])/2)) 
v['c'] = np.ones(int(len(v['g'])/2)) 
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
    plt.plot((hypdif['c'].real/dif['c'].real), label = '$ \\frac{\\nu_0q^{i - i_0} \\nabla^2}{\\nabla^2} $', lw = 2, color = CB91_Blue)  

else:
    plt.plot(dif['c'].real, label = '$ \\nabla^2 $', lw = 2, color = CB91_Blue)  
    plt.plot(hypdif['c'].real, label = '$ \\nu_0q^{i - i_0} \\nabla^2$', lw = 2, color = CB91_Green)  


plt.legend()
plt.show()



