from firedrake import *
from mpi4py import MPI
import argparse
import pyblockjacobi
from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
from firedrake.solving_utils import _SNESContext
import operator
import numpy

class MatPatch(PCBase):

    def initialize(self, obj):

        if isinstance(obj, PETSc.PC):
            A, P = obj.getOperators()
        else:
            raise ValueError("Not a PC?")

        ctx = get_appctx(obj.getDM())
        if ctx is None:
            raise ValueError("No context found on form")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r", ctx)

        if P.getType() == "python":
            ictx = P.getPythonContext()
            if ictx is None:
                raise ValueError("No context found on matrix")
            if not isinstance(ictx, ImplicitMatrixContext):
                raise ValueError("Don't know how to get form from %r", ictx)
            J = ictx.a
            bcs = ictx.row_bcs
            if bcs != ictx.col_bcs:
                raise NotImplementedError("Row and column bcs must match")
        else:
            J = ctx.Jp or ctx.J
            bcs = ctx._problem.bcs

        mesh = J.ufl_domain()
        self.plex = mesh._plex
        self.ctx = ctx

        if mesh.cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes")

        if "overlap_type" not in mesh._distribution_parameters:
            if mesh.comm.size > 1:
                # Want to do
                # warnings.warn("You almost surely want to set an overlap_type in your mesh's distribution_parameters.")
                # but doesn't warn!
                PETSc.Sys.Print("Warning: you almost surely want to set an overlap_type in your mesh's distribution_parameters.")

        patch = obj.__class__().create(comm=obj.comm)
        patch.setOptionsPrefix(obj.getOptionsPrefix() + "blockjacobi_")
        self.configure_patch(patch, obj)
        patch.setType("blockjacobi")

        Jstate = None

        V, _ = map(operator.methodcaller("function_space"), J.arguments())

        patch.setDM(self.plex)
        self.plex.setDefaultSection(V.dm.getDefaultSection())
        patch.setAttr("ctx", ctx)
        patch.incrementTabLevel(1, parent=obj)
        patch.setFromOptions()
        patch.setUp()
        self.patch = patch

    def update(self, pc):
        self.patch.setUp()

    def view(self, pc, viewer=None):
        self.patch.view(viewer=viewer)

    def configure_patch(self, patch, pc):
        (A, P) = pc.getOperators()
        patch.setOperators(A, P)

    def apply(self, pc, x, y):
        self.patch.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.patch.applyTranspose(x, y)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--pc", type=str, default="matpatch")

args, _ = parser.parse_known_args()
mesh = UnitSquareMesh(args.n, args.n)

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
    "pc_python_type": "__main__.MatPatch",
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
