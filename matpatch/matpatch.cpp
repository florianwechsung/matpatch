#include <iostream>

#include <petsc.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscblaslapack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

class BlockJacobi {
   public:
    vector<vector<PetscInt>> dofsPerBlock;
    vector<vector<PetscScalar>> matValuesPerBlock;
    vector<PetscScalar> work;

    BlockJacobi(vector<vector<PetscInt>> _dofsPerBlock)
        : dofsPerBlock(_dofsPerBlock) {
    
        int numBlocks = dofsPerBlock.size();
        PetscInt dof;
        matValuesPerBlock = vector<vector<PetscScalar>>(numBlocks);
        int biggestBlock = 0;
        for(int p=0; p<numBlocks; p++) {
            dof = dofsPerBlock[p].size();
            matValuesPerBlock[p] = vector<PetscScalar>(dof * dof);
            biggestBlock = max(biggestBlock, dof);
        }
        work = vector<PetscScalar>(biggestBlock, 0);
    }

    PetscInt updateValuesPerBlock(Mat P) {
        int numBlocks = dofsPerBlock.size();
        PetscInt ierr, dof;
        for(int p=0; p<numBlocks; p++) {
            dof = dofsPerBlock[p].size();
            ierr = MatGetValues(P, dof, &dofsPerBlock[p][0], dof, &dofsPerBlock[p][0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);
        }
        for(int p=0; p<numBlocks; p++) {
            PetscInt dof = dofsPerBlock[p].size();
            PetscInt lda=dof;
            vector<PetscInt> piv(dof, 0.);
            vector<PetscScalar> fwork(dof, 0.);
            PetscInt info;
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&dof,&dof,&matValuesPerBlock[p][0],&lda,&piv[0],&info));
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&dof,&matValuesPerBlock[p][0], &lda, &piv[0],&fwork[0],&dof,&info));
        }
        return 0;
    }


    void solve(double* b, double* x) {
        PetscInt dof;
        for(int p=0; p<dofsPerBlock.size(); p++) {
            dof = dofsPerBlock[p].size();
            for(int j=0; j<dof; j++) {
                work[j] = b[dofsPerBlock[p][j]];;
            }
            for(int i=0; i<dof; i++) {
                for(int j=0; j<dof; j++) {
                    x[dofsPerBlock[p][i]] += matValuesPerBlock[p][i*dof + j] * work[j];
                }
            }
        }
    }
};


PetscErrorCode PCSetup_MatPatch(PC pc) {
    auto P = pc -> pmat;
    if(!(pc->data)) {
        PetscInt ierr, i, j, k, p;

        DM  dm, plex;
        ierr = PCGetDM(pc, &dm); CHKERRQ(ierr);
        ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);

        PetscInt vlo, vhi;
        ierr = DMPlexGetDepthStratum(plex, 0, &vlo, &vhi); CHKERRQ(ierr);

        vector<vector<PetscInt>> pointsPerBlock(vhi-vlo);
        PetscInt      *star = NULL, *closure = NULL;
        PetscInt starSize;
        for(p=0; p<vhi-vlo; p++) {
            ierr = DMPlexGetTransitiveClosure(dm, p+vlo, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
            pointsPerBlock[p] = vector<PetscInt>(starSize);
            for(j=0; j<starSize; j++)
                pointsPerBlock[p][j] = star[2*j];
            ierr = DMPlexRestoreTransitiveClosure(dm,  p+vlo, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
        }
        PetscSection dofSection;
        ierr = DMGetDefaultSection(dm, &dofSection);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) dofSection);CHKERRQ(ierr);

        vector<vector<PetscInt>> dofsPerBlock(vhi-vlo);
        PetscInt dof, off, numDofs;
        PetscInt blocksize = 1;
        MatGetBlockSize(P, &blocksize);
        for(p=0; p<vhi-vlo; p++) {
            numDofs = 0;
            for(i=0; i<pointsPerBlock[p].size(); i++) {
                ierr = PetscSectionGetDof(dofSection, pointsPerBlock[p][i], &dof);CHKERRQ(ierr);
                numDofs += dof * blocksize;
            }
            dofsPerBlock[p] = vector<PetscInt>();
            dofsPerBlock[p].reserve(numDofs);
            for(i=0; i<pointsPerBlock[p].size(); i++) {
                ierr = PetscSectionGetDof(dofSection, pointsPerBlock[p][i], &dof);CHKERRQ(ierr);
                ierr = PetscSectionGetOffset(dofSection, pointsPerBlock[p][i], &off);CHKERRQ(ierr);
                for(j=0; j<dof; j++) {
                    for(k=0; k<blocksize; k++) {
                        dofsPerBlock[p].push_back(k + blocksize * (off + j));
                    }
                }
            }
        }
        if(0) {
            cout << "Points " << endl;
            for(i=0; i<pointsPerBlock.size(); i++) {
                cout << "Block " << i << endl << "\t";
                for(j=0; j<pointsPerBlock[i].size(); j++) {
                    cout << pointsPerBlock[i][j] << " ";
                }
                cout << endl;
            }
            cout << "Dofs " << endl;
            for(i=0; i<dofsPerBlock.size(); i++) {
                cout << "Block " << i << endl << "\t";
                for(j=0; j<dofsPerBlock[i].size(); j++) {
                    cout << dofsPerBlock[i][j] << " ";
                }
                cout << endl;
            }
        }
        auto blockjacobi = new BlockJacobi(dofsPerBlock);
        pc->data = (void *)blockjacobi;
    }
    auto blockjacobi = (BlockJacobi *)pc->data;
    blockjacobi -> updateValuesPerBlock(P);
    return 0;
}

PetscErrorCode PCApply_MatPatch(PC pc, Vec b, Vec x) {
    auto blockjacobi = (BlockJacobi *)pc->data;

    //PetscInt size;
    //VecGetSize(b, &size);
    //cout << "Size: " << size << endl;
    VecSet(x, 0.0);

    double *barray, *xarray;
    VecGetArray(b, &barray);
    VecGetArray(x, &xarray);
    blockjacobi->solve(barray, xarray);
    VecRestoreArray(b, &barray);
    VecRestoreArray(x, &xarray);
    return 0;
}

PetscErrorCode PCDestroy_MatPatch(PC pc) {
    if(pc->data)
        delete (BlockJacobi *)pc->data;
    return 0;
}

PetscErrorCode PCCreate_MatPatch(PC pc) {
    pc->data = NULL;
    pc->ops->apply = PCApply_MatPatch;
    pc->ops->setup = PCSetup_MatPatch;
    pc->ops->destroy = PCDestroy_MatPatch;
    return 0;
}

PYBIND11_MODULE(_matpatch, m) {
    PCRegister("matpatch", PCCreate_MatPatch);
}
