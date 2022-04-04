"""
dXt = b(t,X_t)dt + sigma(t,X_t)dWt
b(t,Xt)  = tanh(Xt)
sigma(t,Xt)=1
a:= 1/2 * sigma * sigma^T
A^* q = \nabla \cdot (b*q) + \nabla^2 * (a*q)

pde: \partial_t q = A^* q
"""

import matplotlib.pyplot as plt
import numpy as np
from pde import DiffusionPDE, ScalarField, UnitGrid, CartesianGrid, PDE

grid = CartesianGrid([[0,4]], [65])
data = np.zeros(65)
data[16]=1  # init condition is delta function
state = ScalarField(grid,data=data) # generate initial condition



eq = PDE({"q":"-d_dx(q*tanh(x)) + laplace(0.5*q)"})
result = eq.solve(state, t_range=0.5)
result.plot()
plt.show()