/***************************************************************
FronTier is a set of libraries that implements differnt types of 
Front Traking algorithms. Front Tracking is a numerical method for 
the solution of partial differential equations whose solutions have 
discontinuities.  

Copyright (C) 1999 by The University at Stony Brook. 

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
****************************************************************/

/*		PETSc.c
 *  Only for one node.
 *      This class PETSc is created to be used a handy interface
 *  to the function calls to PETSc. For each algebric equation 
 *  Ax=b, one PETSc instance is needed, since this instance has 
 *  the storage of these three variables. 
*/ 
#include "solver.h"
#include "petscmat.h"
#include "petscvec.h"     // VLM 
#include "petscdevice.h"  // VLM 
#include "mpi.h"

PETSc::PETSc()
{
        // VLM
        PetscLogEvent  USER_EVENT;
        PetscClassId   classid;
        PetscLogDouble user_event_flops;
        PetscClassIdRegister("class PETSc()",&classid);
        PetscLogEventRegister("User event PETSc()",classid,&USER_EVENT);
        PetscLogEventBegin(USER_EVENT,0,0,0,0);
        

	x = NULL;			/* approx solution, RHS*/
	b = NULL;
  	A = NULL;            		/* linear system matrix */
  	
  	ksp = NULL;        		/* Krylov subspace method context */
	nullsp = NULL;
	pc = NULL;

	Set_petsc_input();  // VLM
	
	KSPCreate(PETSC_COMM_WORLD,&ksp);

	printf("petsc_cuda %d -- in PETSc()\n", petsc_cuda);
	
        // VLM
        PetscLogFlops(user_event_flops);
        PetscLogEventEnd(USER_EVENT,0,0,0,0);
}

PETSc::PETSc(int ilower, int iupper, int d_nz, int o_nz)
{	
        // VLM 
        PetscLogEvent  USER_EVENT;
        PetscClassId   classid;
        PetscLogDouble user_event_flops;
        PetscClassIdRegister("class PETSc(.)",&classid);
        PetscLogEventRegister("User event PETSc(.)",classid,&USER_EVENT);
        PetscLogEventBegin(USER_EVENT,0,0,0,0);
        

	x = NULL;      			/* approx solution, RHS*/
	b = NULL;
  	A = NULL;            		/* linear system matrix */
  	
  	ksp = NULL;          		/* Krylov subspace method context */
	nullsp = NULL;
	pc = NULL;

	Set_petsc_input();  // VLM

	Create(ilower, iupper, d_nz, o_nz);	
	KSPCreate(PETSC_COMM_WORLD,&ksp);

	printf("petsc_cuda %d -- in PETSc(...)\n", petsc_cuda);
	
        // VLM
        PetscLogFlops(user_event_flops);
        PetscLogEventEnd(USER_EVENT,0,0,0,0);
}

void PETSc::Create(int ilower, int iupper, int d_nz, int o_nz)
{	
	Create(PETSC_COMM_WORLD, ilower, iupper, d_nz, o_nz);	
}

void PETSc::Create(
	MPI_Comm Comm, 
	int ilower, 
	int iupper, 
	int d_nz, 
	int o_nz)
{	
	int n	= iupper - ilower +1;
	
	comm 	= Comm;
	iLower	= ilower;	
	iUpper 	= iupper;	
	


        // VLM 
        PetscLogEvent  USER_EVENT;
        PetscClassId   classid;
        PetscLogDouble user_event_flops;
        PetscClassIdRegister("class Create()",&classid);
        PetscLogEventRegister("User event Create ",classid,&USER_EVENT);
        PetscLogEventBegin(USER_EVENT,0,0,0,0);

	printf("petsc_cuda %d -- in Create()\n", petsc_cuda); // VLM 
        

	MatCreateAIJ(Comm,n,n,PETSC_DECIDE,PETSC_DECIDE,
	    d_nz,PETSC_NULL,o_nz,PETSC_NULL,&A);	
	ierr = PetscObjectSetName((PetscObject) A, "A");
	ierr = MatSetFromOptions(A);		
	    
	// VLM: lets see, because of this error: ...  
	// [1]PETSC ERROR: --------------------- Error Message --------------------------------------------------------------
	// [1]PETSC ERROR: Object is in wrong state
	// [1]PETSC ERROR: Must call MatXXXSetPreallocation(), MatSetUp() or the matrix has not yet been factored on argument 1 "mat" 
	// before MatZeroEntries()
	ierr = MatMPIAIJSetPreallocation(A, d_nz, PETSC_NULL, o_nz, PETSC_NULL);
        printf("VLM -- ierr from MatMPIAIJSetPreallocation %d\n",ierr);
	    
	// b
	ierr = VecCreate(PETSC_COMM_WORLD, &b);	
	ierr = PetscObjectSetName((PetscObject) b, "b");
	ierr = VecSetSizes(b, n, PETSC_DECIDE);	
	ierr = VecSetFromOptions(b);
	
	ierr = VecCreate(PETSC_COMM_WORLD,&x);
	ierr = PetscObjectSetName((PetscObject) x, "X");
	ierr = VecSetSizes(x, n, PETSC_DECIDE);	
	ierr = VecSetFromOptions(x);

        // VLM 
        PetscLogFlops(user_event_flops);
        PetscLogEventEnd(USER_EVENT,0,0,0,0);
}

PETSc::~PETSc()
{
	if(x!=NULL)
	{
		VecDestroy(&x);
		x = NULL;
	}
	if(b!=NULL)
	{
		VecDestroy(&b);
		b = NULL;
	}
	if(A!=NULL)
	{
		MatDestroy(&A);
		A = NULL;
	}
	if(ksp!=NULL)
	{
		KSPDestroy(&ksp);
		ksp = NULL;
	}
	if(nullsp!=NULL)
	{
		MatNullSpaceDestroy(&nullsp);
		nullsp = NULL;
	}
}

void PETSc::Set_petsc_input()	// VLM 
{
	extern char *in_name;
	char in_val[100];
	int retv;

	FILE *infile = fopen(in_name,"r");

	petsc_solver_first_time = 1;

	petsc_cuda = 0;
	petsc_hypre_pc = 0;
	strcpy(hypre_thres, "0.5");
	pureNeumann_gmres = 0;
	nonpureNeumann_gmres = 0;
	petsc_use_pcgamg = 0;

	if (CursorAfterStringOpt(infile,"Enter PETSc cuda flag (0/1):"))
	{
	    fscanf(infile, "%s", in_val);
	    petsc_cuda = atoi(in_val); 
	}

	if (CursorAfterStringOpt(infile,"Enter PETSc Hypre preconditioner flag (0/1):"))
	{
	    fscanf(infile, "%s", in_val);
	    petsc_hypre_pc = atoi(in_val); 
	}

        if (CursorAfterStringOpt(infile,"Enter pc_hypre_boomeramg_strong_threshold:"))
	{
            fscanf(infile,"%s", hypre_thres);
	}

	if (CursorAfterStringOpt(infile,"Use GMRES for pure Neumman solver (0/1):"))
	{
	    fscanf(infile, "%s", in_val);
	    pureNeumann_gmres = atoi(in_val); 
	}

	if (CursorAfterStringOpt(infile,"Use GMRES for non pure Neumman solver (0/1):"))
	{
	    fscanf(infile, "%s", in_val);
	    nonpureNeumann_gmres = atoi(in_val); 
	}

	if (CursorAfterStringOpt(infile,"Use PCGAMG as preconditioner for GMRES (0/1):"))
	{
	    fscanf(infile, "%s", in_val);
	    petsc_use_pcgamg = atoi(in_val); 
	}


	// use PETSc's GAMG preconditioner if using the cuda enabled functionality
	// since I ran into some problem with the hypre one and did not have time to figure out why
	if (petsc_cuda)  
	{
	    petsc_hypre_pc = 0;
	}

	retv = fclose(infile);
	printf("\npetsc_hypre_pc %d; pc_hypre_boomeramg_strong_threshold [%s]\n", petsc_hypre_pc, hypre_thres);
	printf("pureNeumann_gmres %d; nonpureNeumann_gmres %d\n", pureNeumann_gmres, nonpureNeumann_gmres);
	printf("petsc_use_pcamg %d\n", petsc_use_pcgamg);
	printf("petsc_solver_first_time %d\n", petsc_solver_first_time);
	printf("petsc_cuda %d; fclose retval %d\n", petsc_cuda, retv);
}

void PETSc::Reset_A()	// Reset all entries to zero ;
{
	MatZeroEntries(A);
}
void PETSc::Reset_b()  //  Reset all entries to zero ;
{
        VecZeroEntries(b);
}
void PETSc::Reset_x()
{
        VecZeroEntries(x);
}

// A
void PETSc::Set_A(PetscInt i, PetscInt j, double val)	// A[i][j]=val;
{
	ierr = MatSetValues(A,1,&i,1,&j,&val,INSERT_VALUES);
}

void PETSc::Add_A(PetscInt i, PetscInt j, double val)	// A[i][j]+=val;
{	
	ierr = MatSetValues(A,1,&i,1,&j,&val,ADD_VALUES);
}

void PETSc::Get_row_of_A(PetscInt i, PetscInt *ncol, PetscInt **cols, double **row)
{	
	ierr = MatGetRow(A,i,ncol,(const PetscInt**)cols,
			(const PetscScalar**)row);
	ierr = MatRestoreRow(A,i,ncol,(const PetscInt**)cols,
			(const PetscScalar**)row);
}

// x
void PETSc::Set_x(PetscInt i, double val)	// x[i]=val;
{
	ierr = VecSetValues(x,1,&i,&val,INSERT_VALUES);	
}

void PETSc::Add_x(PetscInt i, double val)	// x[i]+=val;
{
	ierr = VecSetValues(x,1,&i,&val,ADD_VALUES);
}

void PETSc::Set_b(PetscInt i, double val)	// x[i]=val;
{
	ierr = VecSetValues(b,1,&i,&val,INSERT_VALUES);
}

void PETSc::Add_b(
	PetscInt i, 
	double val)	// x[i]+=val;
{
	ierr = VecSetValues(b,1,&i,&val,ADD_VALUES);
}

void PETSc::Get_x(double *p)
{
	PetscScalar      *values;
	VecGetArray(x,&values);
	for(int i = 0; i < iUpper-iLower+1; i++)
		p[i] = values[i];	
        VecRestoreArray(x,&values); 
}

void PETSc::Get_b(double *p)
{
	PetscScalar      *values;
	VecGetArray(b,&values);
	for(int i = 0; i < iUpper-iLower+1; i++)
		p[i] = values[i];	
        VecRestoreArray(b,&values); 
}

void PETSc::Get_x(double *p, 
	int n, 
	int *global_index)
{
}

void PETSc::SetMaxIter(int val)
{
	PetscInt maxits;
	double rtol, atol, dtol;
	
	KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits);
	ierr = KSPSetTolerances(ksp, rtol, atol, dtol, val);
}	/* end SetMaxIter */

void PETSc::SetTol(double val)
{
	PetscInt maxits;
	double rtol, atol, dtol;
	
	KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits);
	ierr = KSPSetTolerances(ksp, val, atol, dtol, maxits);
}

void PETSc::SetKDim(int val)
{
	
}

void PETSc::GetNumIterations(PetscInt *num_iterations)
{
	KSPGetIterationNumber(ksp,num_iterations);        
}	/* end GetNumIterations */

void PETSc::GetFinalRelativeResidualNorm(double *rel_resid_norm)
{
	KSPGetResidualNorm(ksp,rel_resid_norm);
}	/* end GetFinalRelativeResidualNorm */

void PETSc::Solve_GMRES(void)
{
	// VLM 
	PetscLogEvent  USER_EVENT;
	PetscClassId   classid;
	PetscLogDouble user_event_flops;
	PetscClassIdRegister("class Solve_GMRES()",&classid);
	PetscLogEventRegister("User event Solve_GMRES()",classid,&USER_EVENT);
	PetscLogEventBegin(USER_EVENT,0,0,0,0);
	
        
        start_clock("Assemble matrix -- Solve_GMRES");
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        stop_clock("Assemble matrix -- Solve_GMRES");
  	
        start_clock("Assemble vector x -- Solve_GMRES");
  	ierr = VecAssemblyBegin(x);
  	ierr = VecAssemblyEnd(x);
        stop_clock("Assemble vector x -- Solve_GMRES");
  	
        start_clock("Assemble vector b -- Solve_GMRES");
  	ierr = VecAssemblyBegin(b);
  	ierr = VecAssemblyEnd(b);
	stop_clock("Assembly vector b -- Solve_GMRES");

	start_clock("KSPSetOperators -- Solve_GMRES");
        KSPSetOperators(ksp,A,A);
	stop_clock("KSPSetOperators -- Solve_GMRES");

        // VLM
	PetscReal normx_0;
	VecNorm(x, NORM_2, &normx_0);

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

	    start_clock("KSPSetType -- Solve_GMRES");
	    KSPSetType(ksp,KSPGMRES);
	    stop_clock("KSPSetType -- Solve_GMRES");

	    //start_clock("KSPSetFromOptions -- Solve_GMRES");
            //KSPSetFromOptions(ksp);
	    //stop_clock("KSPSetFromOptions -- Solve_GMRES");

	    start_clock("KSPSetUp -- Solve_GMRES");
            KSPSetUp(ksp);
	    stop_clock("KSPSetUp -- Solve_GMRES");
	}

	start_clock("KSPSolve -- Solve_GMRES");
        KSPSolve(ksp,b,x);
	stop_clock("KSPSolve -- Solve_GMRES");

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d -- Solve_GMRES()\n", reason);
	PetscInt its, itstot;
        KSPGetIterationNumber(ksp, &its);
        KSPGetTotalIterations(ksp, &itstot);
        printf("VLM: KSP iterationNumber, TotalIterations: %d, %d  -- Solve_GMRES()\n", its, itstot);
	PetscReal normx_1;
	VecNorm(x, NORM_2, &normx_1);
	printf("VLM: initial norm x: %e;  final norm x: %e;  difference: % e -- Solve_GMRES() \n", normx_0, normx_1, normx_0 - normx_1);

	// VLM
	PetscLogFlops(user_event_flops);
	PetscLogEventEnd(USER_EVENT,0,0,0,0);

}	/* end Solve_GMRES */

void PETSc::Solve(void)
{
	// VLM
        printf("VLM: Enter PETSC::Solve()\n");
	PetscLogEvent  USER_EVENT;
	PetscClassId   classid;
	PetscLogDouble user_event_flops;
	PetscClassIdRegister("class Solve()",&classid);
	PetscLogEventRegister("User event Solve()",classid,&USER_EVENT);
	PetscLogEventBegin(USER_EVENT,0,0,0,0);
	
#if defined __HYPRE__
	if (petsc_hypre_pc)
	{
	    Solve_HYPRE();
	}
	else
	{
	    if (nonpureNeumann_gmres)
	    {
	        Solve_GMRES();
	    }
	    else
	    {
	        Solve_BCGSL();
	    }
        }
#else // defined __HYPRE__*/
	if (nonpureNeumann_gmres)
	{
	    Solve_GMRES();
	}
	else
	{
	    Solve_BCGSL();
	}
#endif // defined __HYPRE__
	
 	// VLM
	PetscLogFlops(user_event_flops);
	PetscLogEventEnd(USER_EVENT,0,0,0,0);
	
        // VLM 
        //https://petsc.org/main/docs/manualpages/KSP/KSPConvergedReason/
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d; Leave PETSC::Solve()\n", reason);
	PetscInt its, itstot;
        KSPGetIterationNumber(ksp, &its);
        KSPGetTotalIterations(ksp, &itstot);
        printf("VLM: KSP iterationNumber, TotalIterations: %d, %d; Leave PETSC::Solve()\n", its, itstot);
}	/* end Solve */

void PETSc::Solve_BCGSL(void)
{
        
	printf("VLM: Entering Solve_BCGSL()\n");

        start_clock("Assemble matrix -- Solve_BCGSL");
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
	stop_clock("Assembly matrix -- Solve_BCGSL");
  	
        start_clock("Assemble vector x -- Solve_BCGSL");
  	ierr = VecAssemblyBegin(x);
  	ierr = VecAssemblyEnd(x);
	stop_clock("Assembly vector x -- Solve_BCGSL");
  	
        start_clock("Assemble vector b -- Solve_BCGSL");
  	ierr = VecAssemblyBegin(b);
  	ierr = VecAssemblyEnd(b);
	stop_clock("Assembly vector b -- Solve_BCGSL");

        start_clock("KSPSetOperators -- Solve_BCGSL");
        KSPSetOperators(ksp,A,A);
        stop_clock("KSPSetOperators -- Solve_BCGSL");

	// VLM
        PetscReal normx_0;
	VecNorm(x, NORM_2, &normx_0);

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            start_clock("KSPSetType -- Solve_BCGSL");
            KSPSetType(ksp,KSPBCGSL);
            stop_clock("KSPSetType -- Solve_BCGSL");
            start_clock("KSPBCGSLSetEll -- Solve_BCGSL");
	    KSPBCGSLSetEll(ksp,2);
            stop_clock("KSPBCGSLSetEll -- Solve_BCGSL");

            //start_clock("KSPSetFromOptions -- Solve_BCGSL");
            //KSPSetFromOptions(ksp);
            //stop_clock("KSPSetFromOptions -- Solve_BCGSL");

            start_clock("KSPSetUp -- Solve_BCGSL");
            KSPSetUp(ksp);
            stop_clock("KSPSetUp -- Solve_BCGSL");
	}

	start_clock("KSPSolve -- Solve_BCGSL");
        KSPSolve(ksp,b,x);
	stop_clock("KSPSolve -- Solve_BCGSL");

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d -- Solve_BCGSL()\n", reason);
        PetscInt its, itstot;
	KSPGetIterationNumber(ksp, &its);
	KSPGetTotalIterations(ksp, &itstot);
	printf("VLM: KSP iterationNumber, TotalIterations: %d, %d  -- Solve_BCGSL()\n", its, itstot);
	PetscReal normx_1;
	VecNorm(x, NORM_2, &normx_1);
	printf("VLM: initial norm x: %e;  final norm x: %e;  difference: % e -- Solve_BCGSL() \n", normx_0, normx_1, normx_0 - normx_1);
}

void PETSc::Solve_withPureNeumann_GMRES(void)
{
	PC pc;

	if (debugging("trace"))
	    printf("Entering Solve_withPureNeumann_GMRES()\n");

        start_clock("Assemble matrix -- Solve_withPureNeumann_GMRES");
    	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        stop_clock("Assemble matrix -- Solve_withPureNeumann_GMRES");
  	
        start_clock("Assemble vector x -- Solve_withPureNeumann_GMRES");
  	ierr = VecAssemblyBegin(x);
  	ierr = VecAssemblyEnd(x);
        stop_clock("Assemble vector x -- Solve_withPureNeumann_GMRES");

	//VLM 
	PetscReal normx_0;
	VecNorm(x, NORM_2, &normx_0);
  	
        start_clock("Assemble vector b -- Solve_withPureNeumann_GMRES");
  	ierr = VecAssemblyBegin(b);
  	ierr = VecAssemblyEnd(b);
        stop_clock("Assemble vector b -- Solve_withPureNeumann_GMRES");
  	
	
        start_clock("Matrix Null Space -- Solve_withPureNeumann_GMRES");
	MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,PETSC_NULL,&nullsp);
        MatSetNullSpace(A,nullsp);
	MatNullSpaceRemove(nullsp,b);
        stop_clock("Matrix Null Space -- Solve_withPureNeumann_GMRES");

        start_clock("KSPSetOperators -- Solve_withPureNeumann_GMRES");
        KSPSetOperators(ksp,A,A);
        stop_clock("KSPSetOperators -- Solve_withPureNeumann_GMRES");

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            if (petsc_use_pcgamg)
	    {
                start_clock("PCG in pure neumann solver -- Solve_withPureNeumann_GMRES");
                KSPGetPC(ksp,&pc);
                PCSetType(pc,PCGAMG);
                stop_clock("PCG in pure neumann solver -- Solve_withPureNeumann_GMRES");
	    }

            start_clock("KSPSetType -- Solve_withPureNeumann_GMRES");
	    KSPSetType(ksp,KSPGMRES);
            stop_clock("KSPSetType -- Solve_withPureNeumann_GMRES");
            start_clock("KSPSetFromOptions -- Solve_withPureNeumann_GMRES");
            KSPSetFromOptions(ksp);
            stop_clock("KSPFromOptions -- Solve_withPureNeumann_GMRES");

	    start_clock("KSPSetUp in pure neumann solver -- Solve_withPureNeumann_GMRES");
            KSPSetUp(ksp);
	    stop_clock("KSPSetUp in pure neumann solver -- Solve_withPureNeumann_GMRES");
	    start_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_GMRES");
            KSPSolve(ksp,b,x);
	    stop_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_GMRES");
	}

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d -- Solve_withPureNeumann_GMRES()\n", reason);
	PetscInt its, itstot;
        KSPGetIterationNumber(ksp, &its);
        KSPGetTotalIterations(ksp, &itstot);
        printf("VLM: KSP iterationNumber, TotalIterations: %d, %d -- Solve_withPureNeumann_GMRES() \n", its, itstot);
	PetscReal normx_1;
	VecNorm(x, NORM_2, &normx_1);
        printf("VLM: initial norm x: %e;  final norm x: %e;  difference: % e -- Solve_withPureNeumann_GMRES() \n", normx_0, normx_1, normx_0 - normx_1);

	printf("Leaving Solve_withPureNeumann_GMRES()\n");
}	/* end Solve_withPureNeumann_GMRES */

void PETSc::Solve_withPureNeumann(void)
{
	if (petsc_hypre_pc)
	{
	    Solve_withPureNeumann_HYPRE();
	}
	else
	{
	    if (pureNeumann_gmres)
	    {
	        Solve_withPureNeumann_GMRES();
	    }
	    else
	    {
	        Solve_withPureNeumann_BCGSL();   
	    }
	}

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d; Leaving Solve_withPureNeumann\n", reason);
	PetscInt its, itstot;
        KSPGetIterationNumber(ksp, &its);
        KSPGetTotalIterations(ksp, &itstot);
        printf("VLM: KSP iterationNumber, TotalIterations: %d, %d; Leaving Solve_withPureNeumann\n", its, itstot);

}	/* end Solve_withPureNeumann */

void PETSc::Solve_withPureNeumann_HYPRE(void)
{
        PC pc; 

	//VLM
	if (petsc_cuda || !petsc_hypre_pc)
        {
	    printf("VLM: Solve_HYPRE() --> I SHOULDN'T BE HERE!!!!!!\n");
	}
        printf("Entering Solve_withPureNeumann_HYPRE()\n");
	printf("hypre strong_threshold %s\n", hypre_thres);


	if (debugging("trace"))
        printf("Entering Solve_withPureNeumann_HYPRE()\n");

        start_clock("Assemble matrix -- Solve_withPureNeumann_HYPRE");
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        stop_clock("Assemble matrix -- Solve_withPureNeumann_HYPRE");

        start_clock("Assemble vector x -- Solve_withPureNeumann_HYPRE");
        ierr = VecAssemblyBegin(x);
        ierr = VecAssemblyEnd(x);
        stop_clock("Assemble vector x -- Solve_withPureNeumann_HYPRE");

        start_clock("Assemble vector b -- Solve_withPureNeumann_HYPRE");
        ierr = VecAssemblyBegin(b);
        ierr = VecAssemblyEnd(b);
        stop_clock("Assemble vector b -- Solve_withPureNeumann_HYPRE");


        start_clock("Matrix Null Space -- Solve_withPureNeumann_HYPRE");
        MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,PETSC_NULL,&nullsp);
        MatSetNullSpace(A,nullsp);
        MatNullSpaceRemove(nullsp,b);
        stop_clock("Matrix Null Space -- Solve_withPureNeumann_HYPRE");

        start_clock("KSPSetOperators -- Solve_withPureNeumann_HYPRE");
        KSPSetOperators(ksp,A,A);
        stop_clock("KSPSetOperators -- Solve_withPureNeumann_HYPRE");

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            start_clock("KSPSetType -- Solve_withPureNeumann_HYPRE");
            KSPSetType(ksp,KSPBCGS);
            stop_clock("KSPSetType  -- Solve_withPureNeumann_HYPRE");

	    start_clock("HYPRE preconditioner -- Solve_withPureNeumann_HYPRE");
            KSPGetPC(ksp,&pc);
            PCSetType(pc,PCHYPRE);
            PCHYPRESetType(pc,"boomeramg");
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_strong_threshold", hypre_thres); 
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "pmis"); 
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i"); 
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
	    stop_clock("HYPRE preconditioner -- Solve_withPureNeumann_HYPRE");
	}

        start_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_HYPRE");
        KSPSolve(ksp,b,x);
        stop_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_HYPRE");

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d; Leaving Solve_withPureNeumann_HYPRE()\n", reason);

	if (debugging("trace"))
	printf("Leaving Solve_withPureNeumann_HYPRE()\n");
        //fclose(infile);
}

void PETSc::Solve_withPureNeumann_BCGSL(void)
{
        PC pc;

	printf("Entering Solve_withPureNeumann_BCGSL()\n");

        start_clock("Assemble matrix -- Solve_withPureNeumann_BCGSL");
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        stop_clock("Assemble matrix -- Solve_withPureNeumann_BCGSL");
  	
        start_clock("Assemble vector x -- Solve_withPureNeumann_BCGSL");
  	ierr = VecAssemblyBegin(x);
  	ierr = VecAssemblyEnd(x);
        stop_clock("Assemble vector x -- Solve_withPureNeumann_BCGSL");
  	
        start_clock("Assemble vector b -- Solve_withPureNeumann_BCGSL");
  	ierr = VecAssemblyBegin(b);
  	ierr = VecAssemblyEnd(b);
        stop_clock("Assemble vector b -- Solve_withPureNeumann_BCGSL");
  	
        start_clock("Matrix Null Space -- Solve_withPureNeumann_BCGSL");
	MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,PETSC_NULL,&nullsp);
        MatSetNullSpace(A,nullsp);
	MatNullSpaceRemove(nullsp,b);
        stop_clock("Matrix Null Space -- Solve_withPureNeumann_BCGSL");
	
        start_clock("KSPSetOperators -- Solve_withPureNeumann_BCGSL");
        KSPSetOperators(ksp,A,A);
        stop_clock("KSPSetOperators -- Solve_withPureNeumann_BCGSL");

	// VLM
	PetscReal normx_0;
	VecNorm(x, NORM_2, &normx_0);

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            start_clock("PCG in pure neumann solver -- Solve_withPureNeumann_BCGSL");
            KSPGetPC(ksp,&pc);
            PCSetType(pc,PCGAMG);
            stop_clock("PCG in pure neumann solver -- Solve_withPureNeumann_BCGSL");

            start_clock("KSPSetType -- Solve_withPureNeumann_BCGSL");
	    KSPSetType(ksp,KSPBCGSL);
            stop_clock("KSPSetType -- Solve_withPureNeumann_BCGSL");
            start_clock("KSPBCGSLSetEll -- Solve_withPureNeumann_BCGSL");
	    KSPBCGSLSetEll(ksp,2);
            stop_clock("KSPBCGSLSetEll -- Solve_withPureNeumann_BCGSL");

            start_clock("KSPSetFromOptions -- Solve_withPureNeumann_BCGSL");
            KSPSetFromOptions(ksp);
            stop_clock("KSPSetFromOptions -- Solve_withPureNeumann_BCGSL");
            start_clock("KSPSetUp -- Solve_withPureNeumann_BCGSL");
            KSPSetUp(ksp);
            stop_clock("KSPSetUp -- Solve_withPureNeumann_BCGSL");
	}

	start_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_BCGSL");
        KSPSolve(ksp,b,x);
	stop_clock("Petsc Solve in pure neumann solver -- Solve_withPureNeumann_BCGSL");

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d -- Solve_withPureNeumann_BCGSL\n", reason);
	PetscInt its, itstot;
	KSPGetIterationNumber(ksp, &its);
	KSPGetTotalIterations(ksp, &itstot);
	printf("VLM: KSP iterationNumber, TotalIterations: %d, %d  -- Solve_withPureNeumann_BCGSL()\n", its, itstot);
	PetscReal normx_1;
	VecNorm(x, NORM_2, &normx_1);
	printf("VLM: initial norm x: %e;  final norm x: %e;  difference: % e -- Solve_withPureNeumann_BCGSL() \n", normx_0, normx_1, normx_0 - normx_1);

	printf("Leaving Solve_withPureNeumann_BCGSL()\n");
}	/* end Solve_withPureNeumann_BCGSL */

void PETSc::Print_A(const char *filename)
{
	PetscViewer viewer;
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
        MatView(A, viewer);
        PetscViewerDestroy(&viewer);
}	/* end Print_A */

void PETSc::Print_b(const char *filename)
{
        ierr = VecAssemblyBegin(b);
        ierr = VecAssemblyEnd(b);
	//VLM......: PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use PetscViewerPushFormat()/PetscViewerPopFormat()") PetscErrorCode PetscViewerSetFormat(PetscViewer,PetscViewerFormat);
	//PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,
        //			PETSC_VIEWER_ASCII_MATLAB);
	PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,
			PETSC_VIEWER_ASCII_MATLAB);
        VecView(b, PETSC_VIEWER_STDOUT_WORLD);
}	/* end Print_b */

extern void viewTopVariable(
	Front *front,
	double *var,
	boolean set_bounds,
	double var_min,
	double var_max,
	char *dirname,
	char *var_name)
{
	HDF_MOVIE_VAR hdf_movie_var;
	HDF_MOVIE_VAR *hdf_movie_var_save = front->hdf_movie_var;
	front->hdf_movie_var = &hdf_movie_var;
	hdf_movie_var.num_var = 1;
	FT_MatrixMemoryAlloc((POINTER*)&hdf_movie_var.var_name,1,100,
				sizeof(char));
	FT_VectorMemoryAlloc((POINTER*)&hdf_movie_var.top_var,1,
				sizeof(double*));
	FT_VectorMemoryAlloc((POINTER*)&hdf_movie_var.preset_bound,1,
				sizeof(boolean));
	FT_VectorMemoryAlloc((POINTER*)&hdf_movie_var.var_min,1,
				sizeof(double));
	FT_VectorMemoryAlloc((POINTER*)&hdf_movie_var.var_max,1,
				sizeof(double));
	sprintf(hdf_movie_var.var_name[0],"%s",var_name);
	hdf_movie_var.preset_bound[0] = set_bounds;
	hdf_movie_var.var_min[0] = var_min;
	hdf_movie_var.var_max[0] = var_max;
	hdf_movie_var.top_var[0] = var;
	gview_var2d_on_top_grid(front,dirname);

	FT_FreeThese(5,hdf_movie_var.var_name,hdf_movie_var.top_var,
				hdf_movie_var.preset_bound,
				hdf_movie_var.var_min,hdf_movie_var.var_max);
	front->hdf_movie_var = hdf_movie_var_save;
}	/* end viewTopVariable */

#if defined __HYPRE__
void PETSc::Solve_HYPRE(void)
{
        PC pc;

	printf("Entering Solve_HYPRE()\n");

	//VLM
	if (petsc_cuda || !petsc_hypre_pc)
        {
	    printf("VLM: Solve_HYPRE() --> I SHOULDN'T BE HERE!!!!!!\n");
	}
	printf("hypre strong_threshold %s\n", hypre_thres);

        start_clock("Assemble matrix -- Solve_HYPRE");
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        stop_clock("Assembly matrix -- Solve_HYPRE");

        start_clock("Assemble vector x -- Solve_HYPRE");
        ierr = VecAssemblyBegin(x);
        ierr = VecAssemblyEnd(x);
        stop_clock("Assembly vector x -- Solve_HYPRE");

        start_clock("Assemble vector b -- Solve_HYPRE");
        ierr = VecAssemblyBegin(b);
        ierr = VecAssemblyEnd(b);
        stop_clock("Assembly vector b -- Solve_HYPRE");

        start_clock("KSPSetOperators -- Solve_HYPRE");
        KSPSetOperators(ksp,A,A);
        stop_clock("KSPsetOperators -- Solve_HYPRE");

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            start_clock("KSPSetType -- Solve_HYPRE");
	    KSPSetType(ksp,KSPBCGS);
            stop_clock("KSPSetType -- Solve_HYPRE");

	    start_clock("HYPRE preconditioner -- Solve_HYPRE");
            KSPGetPC(ksp,&pc);
	    PCSetType(pc,PCHYPRE);
            PCHYPRESetType(pc,"boomeramg");
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_strong_threshold", hypre_thres);
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "pmis");
            PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i");
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
            stop_clock("HYPRE preconditioner -- Solve_HYPRE"); 
	}

        start_clock("KSPSolve -- Solve_HYPRE");
        KSPSolve(ksp,b,x);
        stop_clock("KSPSolve -- Solve_HYPRE");

        //VLM
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d; Leaving Solve_HYPRE()\n", reason);
}
#endif // defined __HYPRE__

void PETSc::Solve_withPureNeumann_ML(void)
{
	if (debugging("trace"))
	    printf("Entering Solve_withPureNeumann_ML()\n");
	PC pc;
	start_clock("Assemble Matrix in pure neumann solver");
    	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  	
  	ierr = VecAssemblyBegin(x);
  	ierr = VecAssemblyEnd(x);
  	
  	ierr = VecAssemblyBegin(b);
  	ierr = VecAssemblyEnd(b);
	stop_clock("Assemble Matrix in pure neumann solver");
  	
	
	MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,PETSC_NULL,&nullsp);
        MatSetNullSpace(A,nullsp);
	MatNullSpaceRemove(nullsp,b);
	
        KSPSetOperators(ksp,A,A);

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            KSPGetPC(ksp,&pc);
            PCSetType(pc,PCML);

	    KSPSetType(ksp,KSPGMRES);
    
            KSPSetFromOptions(ksp);
	    start_clock("KSP setup in pure neumann solver");
            KSPSetUp(ksp);
	    stop_clock("KSP setup in pure neumann solver");
	}

	start_clock("Petsc Solve in pure neumann solver");
        KSPSolve(ksp,b,x);
	stop_clock("Petsc Solve in pure neumann solver");

        //VLM 
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d\n", reason);

	printf("Leaving Solve_withPureNeumann_ML()\n");
}	/* end Solve_withPureNeumann_ML */


void PETSc::Solve_LU(void)
{
	PC pc;
        start_clock("Assemble matrix and vector");
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

        ierr = VecAssemblyBegin(x);
        ierr = VecAssemblyEnd(x);

        ierr = VecAssemblyBegin(b);
        ierr = VecAssemblyEnd(b);
        stop_clock("Assembly matrix and vector");

        KSPSetOperators(ksp,A,A);

	if (petsc_solver_first_time)  //VLM
	{
            petsc_solver_first_time = 0;

            KSPSetType(ksp,KSPPREONLY);
	    KSPGetPC(ksp,&pc);
	    PCSetType(pc,PCLU);
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
        }

        start_clock("KSPSolve");
        KSPSolve(ksp,b,x);
        stop_clock("KSPSolve");

        //VLM
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        printf("VLM: KSPConvergedReason: %d; Leaving Solve_LU()\n", reason);

} /*direct solver, usually give exact solution for comparison*/
