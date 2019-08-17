from firedrake import *
from mpi4py import MPI
import argparse
from firedrake.petsc import PETSc

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--pc", type=str, default="matpatch")

args, _ = parser.parse_known_args()
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
mesh = UnitSquareMesh(args.n, args.n, distribution_parameters=distribution_parameters)

V = VectorFunctionSpace(mesh, "CG", args.k)

u = Function(V)
v = TestFunction(V)
gamma = Constant(100.0)
rhs = Constant((1, 1))
F = inner(u, v) * dx + inner(grad(u), grad(v)) * dx + gamma * inner(div(u), cell_avg(div(v))) * dx - inner(rhs, v) * dx

common = {
    "snes_type": "ksponly",
    "ksp_type": "cg",
    # "ksp_view": None,
    # "ksp_monitor": None,
    # "ksp_monitor_true_residual": None,
    "ksp_max_it": 1000,
    "ksp_converged_reason": None,
}

matpatch = {
    # "pc_type": "none",
    # "pc_type": "pbjacobi",
    "pc_type": "python",
    "pc_python_type": "matpatch.MatPatch",
    # "pc_python_type": "alfi.Star",
}

patch = {
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    # "patch_pc_patch_sub_mat_type": "aij",
    "patch_pc_patch_sub_mat_type": "dense",
    "patch_pc_patch_multiplicative": False,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_pc_patch_construct_dim": 0,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_pc_patch_construct_type": "star"
}


if args.pc == "matpatch":
    sp = {**common, **matpatch}
elif args.pc == "patch":
    sp = {**common, **patch}
else:
    raise NotImplementedError
bc = DirichletBC(V, 0, "on_boundary")
bc = None
PETSc.Log.begin()
solve(F == 0, u, bcs=bc, solver_parameters=sp)
File("u.pvd").write(u)
comm = mesh.mpi_comm()
events = ["KSPSolve", "PCSetUp", "PCApply"]
perf = dict((e, PETSc.Log.Event(e).getPerfInfo()) for e in events)
perf_reduced = {}
for k, v in perf.items():
    perf_reduced[k] = {}
    for kk, vv in v.items():
        perf_reduced[k][kk] = comm.allreduce(vv, op=MPI.SUM) / comm.size
perf_reduced_sorted = [(k, v) for (k, v) in sorted(perf_reduced.items(), key=lambda d: -d[1]["time"])]
if comm.rank == 0:
    for k, v in perf_reduced_sorted:
        print(GREEN % (("%s:" % k).ljust(30) + "Time = % 6.2fs" % (v["time"])))
        time = perf_reduced_sorted[0][1]["time"]
