#include <iostream>
#include <numeric>
#include <chrono>



#include <petsc.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscblaslapack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <petscpc.h>
#include <petsc4py/petsc4py.h>


using namespace std;
namespace py = pybind11;

PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work);

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
	Mat *localmats;
	vector<IS> dofis;
	vector<PetscInt> piv;
	vector<int> matToUse;
	vector<PetscScalar> fwork;

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
		piv = vector<PetscInt>(biggestBlock, 0.);
		iota(piv.begin(), piv.end(), 1);
		fwork= vector<PetscScalar>(biggestBlock, 0.);
		localmats = NULL;
		dofis = vector<IS>(numBlocks);
		for(int p=0; p<numBlocks; p++) {
			ISCreateGeneral(MPI_COMM_SELF, globalDofsPerBlock[p].size(), &globalDofsPerBlock[p][0], PETSC_USE_POINTER ,&dofis[p]);
		}
        matToUse = vector<int>(numBlocks, -1);
    }

    PetscInt updateValuesPerBlock(Mat P) {
        int numBlocks = dofsPerBlock.size();
        PetscInt ierr, dof;
        if(0) {
            for(int p=0; p<numBlocks; p++) {
                dof = dofsPerBlock[p].size();
                for(int i=0; i<dof; i++)
                    ierr = MatGetValues(P, dof, &dofsPerBlock[p][0], dof, &dofsPerBlock[p][0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);
            }
        } else {
			auto t1 = std::chrono::high_resolution_clock::now();
			ierr = MatCreateSubMatrices(P, numBlocks, &dofis[0], &dofis[0], localmats ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &localmats);CHKERRQ(ierr);
			//ierr = MatCreateSubMatrices(P, numBlocks, &dofis[0], &dofis[0], MAT_INITIAL_MATRIX, &localmats);CHKERRQ(ierr);
            for(int p=0; p<numBlocks; p++) {
                PetscInt dof = globalDofsPerBlock[p].size();
                vector<int> v(dof);
                iota(v.begin(), v.end(), 0);
                ierr = MatGetValues(localmats[p], dof, &v[0], dof, &v[0], &matValuesPerBlock[p][0]);CHKERRQ(ierr);
            }
			//ierr = MatDestroyMatrices(numBlocks, &localmats);CHKERRQ(ierr);
			auto t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
            cout << "Time for getting the local matrices: " << duration << endl << flush;
        }
        if(matToUse[0] == -1) {
            matToUse[0] = 0;
            int savedmats = 0;
            for(int p=1; p<numBlocks; p++) {
                if(matValuesPerBlock[p-1].size() != matValuesPerBlock[p].size()) {
                    matToUse[p] = p;
                    continue;
                }
                int dof = matValuesPerBlock[p-1].size();
                PetscScalar norm = 0;
                for(int i=0; i<dof; i++) {
                    norm += pow(matValuesPerBlock[p][i] - matValuesPerBlock[p-1][i], 2);
                }
                //cout << norm << " ";
                if(norm < 1e-10){
                    savedmats++;
                    matToUse[p] = matToUse[p-1];
                    //matToUse[p] = p;
                } else {
                    matToUse[p] = p;
                }
            }
            cout << "Saved mats: " << savedmats << " out of " << matValuesPerBlock.size() << endl;
        }
		auto t1 = std::chrono::high_resolution_clock::now();
		PetscInt info;
        for(int p=0; p<numBlocks; p++) {
            if(matToUse[p] != p)
                continue;
            PetscInt dof = dofsPerBlock[p].size();
            //PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&dof,&dof,&matValuesPerBlock[p][0],&dof,&piv[0],&info));
            //PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&dof,&matValuesPerBlock[p][0], &dof, &piv[0],&fwork[0],&dof,&info));
			mymatinvert(&dof, &matValuesPerBlock[p][0], &piv[0], &info, &fwork[0]);
        }
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        cout << "Time for factorising the local matrices: " << duration << endl << flush;
        return 0;
    }


    PetscInt solve(double* b, double* x) {
		auto t1 = std::chrono::high_resolution_clock::now();
        PetscInt dof;
        PetscScalar dOne = 1.0;
        PetscInt one = 1;
        PetscScalar dZero = 0.0;
        for(int p=0; p<dofsPerBlock.size(); p++) {
            dof = dofsPerBlock[p].size();
            for(int j=0; j<dof; j++)
                workb[j] = b[dofsPerBlock[p][j]];;
            if(dof < 6)
                for(int i=0; i<dof; i++)
                    for(int j=0; j<dof; j++)
                        x[dofsPerBlock[p][i]] += matValuesPerBlock[matToUse[p]][i*dof + j] * workb[j];
            else {
                PetscStackCallBLAS("BLASgemv",BLASgemv_("N", &dof, &dof, &dOne, &matValuesPerBlock[matToUse[p]][0], &dof, &workb[0], &one, &dZero, &worka[0], &one));
                for(int i=0; i<dof; i++)
                    x[dofsPerBlock[p][i]] += worka[i];
            }
        }
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        //cout << "Time for applying the local matrices: " << duration << endl << flush;
        return 0;
    }
};

PetscErrorCode MakeSF(DM dm, PetscSF sf, int bs, PetscSF *newsf) {
    PetscInt ierr;
	if(bs==1) {
		*newsf=sf;
		return 0;
	}
    PetscInt nroots, nleaves;
    const PetscInt *ilocal;
    const PetscSFNode *iremote;
    ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
    PetscInt newnroots = bs*nroots;
    PetscInt newnleaves = bs*nleaves;
    PetscInt *newilocal;
    PetscSFNode *newiremote;
    ierr = PetscMalloc1(newnleaves, &newilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(newnleaves, &newiremote);CHKERRQ(ierr);
    for(int i=0; i<nleaves; i++) {
        for(int j=0; j<bs; j++) {
            newilocal[bs*i + j] = bs*ilocal[i] + j;
            newiremote[bs*i + j].index = bs * iremote[i].index + j;
            newiremote[bs*i + j].rank = iremote[i].rank;
        }
    }
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), newsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*newsf, newnroots,newnleaves,newilocal, PETSC_OWN_POINTER , newiremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    return 0;
}

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
                for(j=0; j<dof; j++) {
                    for(k=0; k<blocksize; k++) {
                        dofsPerBlock[p].push_back(k + blocksize * (off + j));
                    }
                }
                //std::sort(dofsPerBlock[p].begin(), dofsPerBlock[p].end());
				globalDofsPerBlock[p] = vector<PetscInt>(dofsPerBlock[p].size(), 0);
                ISLocalToGlobalMappingApply(lgr, dofsPerBlock[p].size(), &dofsPerBlock[p][0], &globalDofsPerBlock[p][0]);
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
        PetscSF newsf;
		MakeSF(dm, sf, blocksize, &newsf);
        auto blockjacobi = new BlockJacobi(dofsPerBlock, globalDofsPerBlock, blocksize*localSize, newsf);
        pc->data = (void *)blockjacobi;
    }
    auto blockjacobi = (BlockJacobi *)pc->data;
    blockjacobi -> updateValuesPerBlock(P);
    return 0;
}

PetscErrorCode PCApply_MatPatch(PC pc, Vec b, Vec x) {
    PetscInt ierr;
	auto t1 = std::chrono::high_resolution_clock::now();
	ierr = VecSet(x, 0.0);CHKERRQ(ierr);
	auto blockjacobi = (BlockJacobi *)pc->data;

	const PetscScalar *globalb;
	PetscScalar *globalx;

	ierr = VecGetArrayRead(b, &globalb);CHKERRQ(ierr);
	ierr = PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);
	ierr = PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(b, &globalb);CHKERRQ(ierr);
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    //cout << "Time for getting the local vectors: " << duration << endl << flush;

	for(int i=0; i<blockjacobi->localx.size(); i++)
		blockjacobi->localx[i] = 0.;

	blockjacobi->solve(&(blockjacobi->localb[0]), &(blockjacobi->localx[0]));
	t1 = std::chrono::high_resolution_clock::now();
	ierr = VecGetArray(x, &globalx);CHKERRQ(ierr);
	ierr = PetscSFReduceBegin(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
	ierr = PetscSFReduceEnd(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
	ierr = VecRestoreArray(x, &globalx);CHKERRQ(ierr);
	t2 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    //cout << "Time for scattering the local vectors: " << duration << endl << flush;
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
// pybind11 casters for PETSc/petsc4py objects, copied from dolfinx repo
// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                  \
  if (!func)                                                                   \
  {                                                                            \
    if (import_petsc4py() != 0)                                                \
      throw std::runtime_error("Error when importing petsc4py");               \
  }

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                               \
  template <>                                                                  \
  class type_caster<_p_##TYPE>                                                 \
  {                                                                            \
  public:                                                                      \
    PYBIND11_TYPE_CASTER(TYPE, _(#NAME));                                      \
    bool load(handle src, bool)                                                \
    {                                                                          \
      if (src.is_none())                                                       \
      {                                                                        \
        value = nullptr;                                                       \
        return true;                                                           \
      }                                                                        \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_Get);                                \
      if (PyObject_TypeCheck(src.ptr(), &PyPetsc##P4PYTYPE##_Type) == 0)       \
        return false;                                                          \
      value = PyPetsc##P4PYTYPE##_Get(src.ptr());                              \
      return true;                                                             \
    }                                                                          \
                                                                               \
    static handle cast(TYPE src, pybind11::return_value_policy policy,         \
                       handle parent)                                          \
    {                                                                          \
      VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_New);                                \
      auto obj = PyPetsc##P4PYTYPE##_New(src);                                 \
      if (policy == pybind11::return_value_policy::take_ownership)             \
        PetscObjectDereference((PetscObject)src);                              \
      return pybind11::handle(obj);                                            \
    }                                                                          \
                                                                               \
    operator TYPE() { return value; }                                          \
  }

namespace pybind11
{
    namespace detail
    {
        PETSC_CASTER_MACRO(PC, PC, pc);
        PETSC_CASTER_MACRO(PetscSection, Section, petscsection);
        PETSC_CASTER_MACRO(DM, DM, dm);
    }
}


PYBIND11_MODULE(_matpatch, m) {
	PCRegister("matpatch", PCCreate_MatPatch);
	m.def("DoSomething", [](PetscSection sec) {
            PetscInt size, ierr;
            ierr = PetscSectionGetStorageSize(sec, &size);CHKERRQ(ierr);
            cout << "Section has size " << size << endl;
			return ierr; 
			});
}
