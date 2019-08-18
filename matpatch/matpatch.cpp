#include <iostream>
#include <numeric>


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
    vector<vector<PetscInt>> globalDofsPerBlock;
    vector<vector<PetscScalar>> matValuesPerBlock;
    vector<PetscScalar> worka;
    vector<PetscScalar> workb;
    vector<PetscScalar> localb;
    vector<PetscScalar> localx;
    PetscSF sf;

    BlockJacobi(vector<vector<PetscInt>> _dofsPerBlock, vector<vector<PetscInt>> _globalDofsPerBlock, int localSize, PetscSF _sf)
        : dofsPerBlock(_dofsPerBlock), globalDofsPerBlock(_globalDofsPerBlock), sf(_sf) {
    
        int numBlocks = dofsPerBlock.size();
        PetscInt dof;
        matValuesPerBlock = vector<vector<PetscScalar>>(numBlocks);
        int biggestBlock = 0;
        for(int p=0; p<numBlocks; p++) {
            dof = dofsPerBlock[p].size();
            matValuesPerBlock[p] = vector<PetscScalar>(dof * dof);
            biggestBlock = max(biggestBlock, dof);
        }
        worka = vector<PetscScalar>(biggestBlock, 0);
        workb = vector<PetscScalar>(biggestBlock, 0);
        localb = vector<PetscScalar>(localSize, 0);
        localx = vector<PetscScalar>(localSize, 0);
    }

    PetscInt updateValuesPerBlock(Mat P) {
        int numBlocks = dofsPerBlock.size();
        PetscInt ierr, dof;
        if(0) {
            for(int p=0; p<numBlocks; p++) {
                dof = dofsPerBlock[p].size();
                for(int i=0; i<dof; i++)

                    ierr = MatGetValues(P, dof, &dofsPerBlock[p][0], dof, &dofsPerBlock[p][0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);
                //ierr = MatGetValues(P, dof, &globalDofsPerBlock[p][0], dof, &globalDofsPerBlock[p][0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);

            }
        } else {
            vector<IS> dofis(numBlocks);
            for(int p=0; p<numBlocks; p++) {
                ierr = ISCreateGeneral(MPI_COMM_SELF, globalDofsPerBlock[p].size(), &globalDofsPerBlock[p][0],PETSC_USE_POINTER ,&dofis[p]); CHKERRQ(ierr);
            }
            Mat *localmats;
            ierr = MatCreateSubMatrices(P, numBlocks, &dofis[0], &dofis[0], MAT_INITIAL_MATRIX, &localmats);CHKERRQ(ierr);
            IS temp;
            for(int p=0; p<numBlocks; p++) {
                PetscInt dof = globalDofsPerBlock[p].size();
                vector<int> v(dof);
                iota(v.begin(), v.end(), 0);
                ierr = MatGetValues(localmats[p], dof, &v[0], dof, &v[0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);
            }

        }

        if(0) {
            cout << "Block mats" << endl;
            for(int p=0; p<numBlocks; p++) {
                cout << "Block " << p << endl;
                for(int i=0; i<matValuesPerBlock[p].size(); i++)
                {
                    cout << matValuesPerBlock[p][i] << " ";
                }
                cout << endl << std::flush;;
            }
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
        if(0) {
            cout << "Block inverses" << endl;
            for(int p=0; p<numBlocks; p++) {
                cout << "Block " << p << endl;
                for(int i=0; i<matValuesPerBlock[p].size(); i++)
                {
                    cout << matValuesPerBlock[p][i] << " ";
                }
                cout << endl << std::flush;;
            }
        }

        return 0;
    }


    PetscInt solve(double* b, double* x) {
        PetscInt dof;
        PetscScalar dOne = 1.0;
        PetscInt one = 1;
        PetscScalar dZero = 0.0;
        for(int p=0; p<dofsPerBlock.size(); p++) {
            dof = dofsPerBlock[p].size();
            for(int j=0; j<dof; j++) {
                workb[j] = b[dofsPerBlock[p][j]];;
            }
            if(dof < 7) {
                for(int i=0; i<dof; i++) {
                    for(int j=0; j<dof; j++) {
                        x[dofsPerBlock[p][i]] += matValuesPerBlock[p][i*dof + j] * workb[j];
                    }
                }
            } else {
                PetscStackCallBLAS("BLASgemv",BLASgemv_("N", &dof, &dof, &dOne, &matValuesPerBlock[p][0], &dof, &workb[0], &one, &dZero, &worka[0], &one));
                for(int i=0; i<dof; i++) {
                    x[dofsPerBlock[p][i]] += worka[i];
                }
            }
        }
        return 0;
    }
};


PetscErrorCode PCSetup_MatPatch(PC pc) {
    auto P = pc -> pmat;
    if(!(pc->data)) {
        ISLocalToGlobalMapping lgr;
        ISLocalToGlobalMapping lgc;
        MatGetLocalToGlobalMapping(P, &lgr, &lgc);
        PetscInt ierr, i, j, k, p;

        DM  dm, plex;
        ierr = PCGetDM(pc, &dm); CHKERRQ(ierr);
        ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);

        PetscInt vlo, vhi, pStart, pEnd;
        ierr = DMPlexGetDepthStratum(plex, 0, &vlo, &vhi); CHKERRQ(ierr);
		DMLabel         ghost = NULL;
		PetscBool      flg;
		ierr = DMGetLabel(dm, "pyop2_ghost", &ghost);CHKERRQ(ierr);
		ierr = DMPlexGetChart(plex, &pStart, &pEnd);CHKERRQ(ierr);
		ierr = DMLabelCreateIndex(ghost, pStart, pEnd);CHKERRQ(ierr);
        vector<vector<PetscInt>> pointsPerBlock;
		pointsPerBlock.reserve(vhi-vlo);
        PetscInt      *star = NULL, *closure = NULL;
        PetscInt starSize;
		int numBlocks = 0;
		for(p=0; p<vhi-vlo; p++) {
			ierr = DMLabelHasPoint(ghost, p+vlo, &flg);CHKERRQ(ierr);
			if (flg) continue;
            ierr = DMPlexGetTransitiveClosure(dm, p+vlo, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
            pointsPerBlock.push_back(vector<PetscInt>(starSize));
            for(j=0; j<starSize; j++)
                pointsPerBlock[numBlocks][j] = star[2*j];
			ierr = DMPlexRestoreTransitiveClosure(dm,  p+vlo, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
			numBlocks++;
        }
        PetscSection dofSection;
        ierr = DMGetDefaultSection(dm, &dofSection);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject) dofSection);CHKERRQ(ierr);

        vector<vector<PetscInt>> dofsPerBlock(numBlocks);
        vector<vector<PetscInt>> globalDofsPerBlock(numBlocks);
        PetscInt dof, off, numDofs;
        PetscInt blocksize = 1;
        MatGetBlockSize(P, &blocksize);
        for(p=0; p<pointsPerBlock.size(); p++) {
            numDofs = 0;
            for(i=0; i<pointsPerBlock[p].size(); i++) {
                ierr = PetscSectionGetDof(dofSection, pointsPerBlock[p][i], &dof);CHKERRQ(ierr);
                numDofs += dof * blocksize;
            }
            dofsPerBlock[p] = vector<PetscInt>();
            dofsPerBlock[p].reserve(numDofs);
            globalDofsPerBlock[p] = vector<PetscInt>();
            globalDofsPerBlock[p].reserve(numDofs);
            for(i=0; i<pointsPerBlock[p].size(); i++) {
                ierr = PetscSectionGetDof(dofSection, pointsPerBlock[p][i], &dof);CHKERRQ(ierr);
                ierr = PetscSectionGetOffset(dofSection, pointsPerBlock[p][i], &off);CHKERRQ(ierr);
                vector<PetscInt> globalDofs(dof, 0);
                vector<PetscInt> localDofs(dof, 0);
                for(j=0; j<dof; j++) {
                    localDofs[j] = off + j;
                }
                ISLocalToGlobalMappingApply(lgr, dof, &localDofs[0], &globalDofs[0]);
                for(j=0; j<dof; j++) {
                    for(k=0; k<blocksize; k++) {
                        dofsPerBlock[p].push_back(k + blocksize * (off + j));
                        globalDofsPerBlock[p].push_back(k + blocksize * globalDofs[j]);
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
        }
        if(0) {
            cout << "Dofs " << endl;
            for(i=0; i<dofsPerBlock.size(); i++) {
                cout << "Block " << i << endl << "\t";
                for(j=0; j<dofsPerBlock[i].size(); j++) {
                    cout << dofsPerBlock[i][j] << " ";
                }
                cout << endl;
            }
        }
        if(0) {
            cout << "Global dofs " << endl;
            for(i=0; i<globalDofsPerBlock.size(); i++) {
                cout << "Block " << i << endl << "\t";
                for(j=0; j<globalDofsPerBlock[i].size(); j++) {
                    cout << globalDofsPerBlock[i][j] << " ";
                }
                cout << endl;
            }
        }
        PetscInt localSize;
        ierr = PetscSectionGetStorageSize(dofSection, &localSize);CHKERRQ(ierr);
        PetscSF sf;
        ierr = DMGetDefaultSF(dm, &sf);CHKERRQ(ierr);
        auto blockjacobi = new BlockJacobi(dofsPerBlock, globalDofsPerBlock, blocksize*localSize, sf);
        pc->data = (void *)blockjacobi;
    }
    auto blockjacobi = (BlockJacobi *)pc->data;
    blockjacobi -> updateValuesPerBlock(P);
    return 0;
}

PetscErrorCode PCApply_MatPatch(PC pc, Vec b, Vec x) {
    //PetscInt size;
    //VecGetSize(b, &size);
    //cout << "bsize " << size << endl;
    //VecGetSize(x, &size);
    //cout << "xsize " << size << endl;
    PetscInt ierr;
    auto blockjacobi = (BlockJacobi *)pc->data;

    PetscScalar *globalb;
    PetscScalar *globalx;
    for(int i=0; i<blockjacobi->localx.size(); i++)
        blockjacobi->localb[i] = 0.;
    ierr = VecGetArray(b, &globalb);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);

    //cout << "localb ";
    //for(int i=0; i<blockjacobi->localb.size(); i++) {
    //    cout << blockjacobi->localb[i] << " ";
    //}
    //cout << endl;
    //cout << "globalb ";
    //for(int i=0; i<blockjacobi->localx.size(); i++) {
    //    cout << globalb[i] << " ";
    //}
    //cout << endl;
    ierr = VecRestoreArray(b, &globalb);CHKERRQ(ierr);

    VecSet(x, 0.0);
    for(int i=0; i<blockjacobi->localx.size(); i++)
        blockjacobi->localx[i] = 0.;


    blockjacobi->solve(&(blockjacobi->localb[0]), &(blockjacobi->localx[0]));
    ierr = VecGetArray(x, &globalx);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
    //cout << "globalx ";
    //for(int i=0; i<blockjacobi->localx.size(); i++) {
    //    cout << globalx[i] << " ";
    //}
    ierr = VecRestoreArray(x, &globalx);CHKERRQ(ierr);

    //VecSet(x, 0.0);
    //double *barray, *xarray;
    //VecGetArray(b, &barray);
    //VecGetArray(x, &xarray);
    //blockjacobi->solve(barray, xarray);
    ////cout << "xarray ";
    ////for(int i=0; i<blockjacobi->localx.size(); i++) {
    ////    cout << xarray[i] << " ";
    ////}
    //VecRestoreArray(b, &barray);
    //VecRestoreArray(x, &xarray);
    return 0;
}

PetscErrorCode PCDestroy_MatPatch(PC pc) {
    if(pc->data)
        delete (BlockJacobi *)pc->data;
    return 0;
}

PetscErrorCode PCSetSF(PC pc, PetscSF sf) {
    cout << "In PCSetSF" << endl;
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
