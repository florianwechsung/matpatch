from firedrake import *
from firedrake.petsc import PETSc
from matpatch import DoSomething

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
section = V.dm.getDefaultSection()

DoSomething(section)


