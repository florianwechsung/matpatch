from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
from firedrake.solving_utils import _SNESContext
import operator
import numpy
from firedrake import *

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
        patch.setOptionsPrefix(obj.getOptionsPrefix() + "matpatch_")
        self.configure_patch(patch, obj)
        patch.setType("matpatch")

        Jstate = None

        V, _ = map(operator.methodcaller("function_space"), J.arguments())

        patch.setDM(self.plex)
        self.plex.setDefaultSection(V.dm.getDefaultSection())
        self.plex.setDefaultGlobalSection(V.dm.getDefaultGlobalSection())
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
