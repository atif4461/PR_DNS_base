/*********************************************************************
FronTier is a set of libraries that implements differnt types of Front 
Traking algorithms. Front Tracking is a numerical method for the solution 
of partial differential equations whose solutions have discontinuities.  


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
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

**********************************************************************/


/*
*				climate.c:
*
*	Copyright 1999 by The University at Stony Brook, All rights reserved.
*
*/

#include "../iFluid/iFluid.h"
#include "../iFluid/ifluid_basic.h"
#include "climate.h"
#include <sys/time.h>
//#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <cuda.h>
#include <iostream>
#include <memory>

#define		MAX_NUM_VERTEX_IN_CELL		20
	/*  Local Application Function Declarations */

static void	melting_flow_driver(Front*,VCARTESIAN*, 
				Incompress_Solver_Smooth_Basis *);
static int      rgbody_vel(POINTER,Front*,POINT*,HYPER_SURF_ELEMENT*,
                        HYPER_SURF*,double*);
static double 	temperature_func(double*,COMPONENT,double);
static void 	read_movie_options(char*,PARAMS*);
static void	melt_flow_point_propagate(Front*,POINTER,POINT*,POINT*,
			HYPER_SURF_ELEMENT*,HYPER_SURF*,double,double*);

extern  char  *in_name;
char *restart_state_name,*restart_name,*out_name2;
boolean RestartRun;
boolean ReadFromInput;
int RestartStep;

int main(int argc, char **argv)
{
#ifdef __PRDNS_TIMER__
        struct timeval tv1,tv2,tv3,tv4;
        gettimeofday(&tv1, NULL);
#endif

	static Front front;
	static F_BASIC_DATA f_basic;
	static LEVEL_FUNC_PACK level_func_pack;
	static VELO_FUNC_PACK velo_func_pack;
	static PARAMS eqn_params;
	static IF_PARAMS iFparams;
	int dim;

	FT_Init(argc,argv,&f_basic);
        PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

	VCARTESIAN *v_cartesian = new VCARTESIAN(front);
	Incompress_Solver_Smooth_Basis *l_cartesian = NULL;
	if (f_basic.dim == 2)
	    l_cartesian = new Incompress_Solver_Smooth_2D_Cartesian(front);
	else if (f_basic.dim == 3)
	    l_cartesian = new Incompress_Solver_Smooth_3D_Cartesian(front);

	/* Initialize basic computational data */
	f_basic.size_of_intfc_state = sizeof(STATE);

	in_name      		= f_basic.in_name;
	restart_state_name      = f_basic.restart_state_name;
        out_name2    		= f_basic.out_name;
        restart_name 		= f_basic.restart_name;
        RestartRun   		= f_basic.RestartRun;
        ReadFromInput   	= f_basic.ReadFromInput;
	RestartStep 		= f_basic.RestartStep;
	dim	 		= f_basic.dim;


	sprintf(restart_state_name,"%s/state.ts%s",restart_name,
			right_flush(RestartStep,7));
	sprintf(restart_name,"%s/intfc-ts%s",restart_name,
			right_flush(RestartStep,7));
	printf("*****************************************\n");
	printf(" DNS of entrainment and mixing: ver 1015\n");
	printf("*****************************************\n");
    
        printf("zhangtao: %d in file %s\n", __LINE__, __FILE__);
        FT_ReadSpaceDomain(in_name,&f_basic);
        printf("zhangtao: %d in file %s\n", __LINE__, __FILE__);

	FT_StartUp(&front,&f_basic);
	FT_InitDebug(in_name);


	eqn_params.dim = f_basic.dim;
	iFparams.dim = f_basic.dim;
	front.extra1 = (POINTER)&iFparams;
	front.extra2 = (POINTER)&eqn_params;
	read_CL_prob_type(&front);
	read_movie_options(in_name,&eqn_params);
	readPhaseParams(&front);
        read_iFparams(in_name,&iFparams);

	if (!RestartRun)
	{
	    if(eqn_params.no_droplets == NO)
	    {
		printf("Initializing droplets\n");
		level_func_pack.pos_component = LIQUID_COMP2;
	        FT_InitIntfc(&front,&level_func_pack);
                initWaterDrops(&front);
	        if (debugging("trace")) printf("Passed init water droplets()\n");
	    }
	    else
	    {
	        printf("No droplets contained\n");
	        level_func_pack.func_params = NULL;
                level_func_pack.func = NULL;
	        level_func_pack.pos_component = LIQUID_COMP2;
	        level_func_pack.wave_type = -1; 
	        FT_InitIntfc(&front,&level_func_pack);
	    }
	    if (f_basic.dim != 3)
                FT_ClipIntfcToSubdomain(&front);
	    if (debugging("trace"))
                printf("Passed FT_ClipIntfcToSubdomain()\n");
	}
	else
	    readWaterDropsParams(&front,restart_state_name);

	FT_ReadTimeControl(in_name,&front);

	/* Initialize velocity field function */

	velo_func_pack.func_params = (POINTER)&iFparams;
	velo_func_pack.func = NULL;
	velo_func_pack.point_propagate = NULL;

	FT_InitVeloFunc(&front,&velo_func_pack);

        v_cartesian->initMesh();
	l_cartesian->initMesh();
	l_cartesian->findStateAtCrossing = ifluid_find_state_at_crossing;
	if (RestartRun)
	{
	    v_cartesian->readFrontInteriorState(restart_state_name);
	    FT_FreeGridIntfc(&front);
	    l_cartesian->readFrontInteriorStates(restart_state_name);
	    /*hook the fields for solvers*/
	    eqn_params.field->vel = iFparams.field->vel;
	    eqn_params.field->pres = iFparams.field->pres;
	    iFparams.field->ext_accel = eqn_params.field->ext_accel;
	}
	else
	{

            init_fluid_state_func(&front,l_cartesian);
            init_vapor_state_func(&front,v_cartesian);
            init_temp_state_func(&front,v_cartesian);

	    if(eqn_params.init_state == FOURIER_STATE)
		    l_cartesian->setParallelVelocity();
	    else if(eqn_params.init_state == FOURIER_STATE_HEFFTE)
		    l_cartesian->setParallelVelocityParallelized();
	    else
	        l_cartesian->setInitialCondition();
            if (debugging("trace"))
                printf("Passed iFluid setInitialCondition()\n");
	    FT_FreeGridIntfc(&front);
	    /*hook the fields for solvers*/
	    eqn_params.field->vel = iFparams.field->vel;
	    eqn_params.field->pres = iFparams.field->pres;

	    v_cartesian->setInitialCondition();
	    iFparams.field->ext_accel = eqn_params.field->ext_accel;
	    if (debugging("trace")) 
                printf("Passed vcartesian setInitialCondition()\n");
	}

	FT_InitVeloFunc(&front,&velo_func_pack);

	if (debugging("trace")) printf("Passed FT_InitVeloFunc()\n");

#ifdef __CUDA__
        initDeviceParticle();
#endif



	FT_SetGlobalIndex(&front);
#ifdef __PRDNS_TIMER__
        gettimeofday(&tv2, NULL);
        printf("\n atif0 Main initialize :  %10.2f", (tv2.tv_usec - tv1.tv_usec)/1000000.0 + (tv2.tv_sec - tv1.tv_sec));
#endif
	/* Propagate the front */
	melting_flow_driver(&front,v_cartesian,l_cartesian);

#ifdef __CUDA__
        clearDeviceParticle();
#endif

#ifdef __PRDNS_TIMER__
        gettimeofday(&tv3, NULL);
#endif
	PetscFinalize();
#ifdef __PRDNS_TIMER__
        gettimeofday(&tv4, NULL);
        printf("atif0 Main Finalize :  %10.2f \n", (tv4.tv_usec - tv3.tv_usec)/1000000.0 + (tv4.tv_sec - tv3.tv_sec));
#endif
	clean_up(0);
}

static  void melting_flow_driver(
        Front *front,
	VCARTESIAN *v_cartesian,
	Incompress_Solver_Smooth_Basis *l_cartesian)
{
        struct timeval tv1,tv2,tv3,tv4,tv5,tv6,tv7,tv8;
#ifdef __PRDNS_TIMER__
        gettimeofday(&tv7, NULL);
#endif

        double CFL;
        int  dim = front->rect_grid->dim;
	IF_PARAMS *iFparams;
	PARAMS *eqn_params;
	MOVIE_OPTION *movie_option;
        double time;
        static LEVEL_FUNC_PACK level_func_pack;
        double runtime, t1(0.);
        double totaltime = 0.0;

	if (debugging("trace"))
	    printf("Entering melting_flow_driver()\n");
	Curve_redistribution_function(front) = expansion_redistribute;
        CFL = Time_step_factor(front);

	iFparams = (IF_PARAMS*)front->extra1;
	eqn_params = (PARAMS*)front->extra2;
	movie_option = eqn_params->movie_option;

	front->hdf_movie_var = NULL;

        if (!RestartRun)
        {
	    FT_ResetTime(front);
            FT_SetOutputCounter(front);
            /* Front standard output*/
 	    /* FT_Save(front,out_name);
            v_cartesian->printFrontInteriorState(out_name);
            l_cartesian->printFrontInteriorStates(out_name);
	    if (eqn_params->prob_type == PARTICLE_TRACKING)
	        printDropletsStates(front,out_name);*/

	    FT_Propagate(front);

	    l_cartesian->solve(front->dt); /*compute pressure for vapor equation*/

	    v_cartesian->solve(front->dt); /*solve vapor equation*/
	    if (debugging("trace"))
	        printf("Solved vapor and temperature\n\n");
	    /*For entrainment problem, droplets in area with supersat > 0*/
	    /*This step must be after one step of v_catesian solver*/
	    if(eqn_params->init_drop_state == PRESET_STATE)
		v_cartesian->initPresetParticles(); 

	    /*For checking the result*/
	    v_cartesian->checkField();
	    printf("Passed checkField()\n");

	    /*Set time step for front*/
	    FT_SetTimeStep(front);
	    l_cartesian->setAdvectionDt();
	    front->dt = std::min(front->dt,CFL*l_cartesian->max_dt);

	    
            l_cartesian->initMovieVariables();
            v_cartesian->initMovieVariables();

            if (eqn_params->prob_type == PARTICLE_TRACKING &&
		movie_option->plot_particles == YES)
	    {
                vtk_plot_scatter(front);
	    }
            FT_AddMovieFrame(front,out_name2,YES);
        }
        else
	{
	    FT_SetOutputCounter(front);
            v_cartesian->initMovieVariables();
            if (eqn_params->prob_type == PARTICLE_TRACKING)
                vtk_plot_scatter(front);
            FT_AddMovieFrame(front,out_name2,YES);
	}

	FT_TimeControlFilter(front);
	/*Record the initial condition*/
	/*v_cartesian->recordField(out_name,"velocity");*/
        if (eqn_params->prob_type == PARTICLE_TRACKING)
	    v_cartesian->output();

#ifdef __CUDA__
        v_cartesian->uploadParticle();
        v_cartesian->initFlg = 0;
#endif
#ifdef __PRDNS_TIMER__
        gettimeofday(&tv8, NULL);
        printf("atif0 Melting flow driver initialize :  %10.2f \n", (tv8.tv_usec - tv7.tv_usec)/1000000.0 + (tv8.tv_sec - tv7.tv_sec));
#endif





      	//std::cout << "CUDA Version: " << CUDA_VERSION / 1000 << (CUDA_VERSION / 10) % 100 << std::endl;
	//torch::Tensor tensor = torch::rand({2, 3});
	//std::cout << tensor << std::endl;
	//
	//torch::jit::script::Module module;
	//try {
	//  // Deserialize the ScriptModule from a file using torch::jit::load().
        //  // module = torch::jit::load("/home/atif/neuraloperator/ico-turb/autoreg5_uv_1000_8_8_100_0.5_0.001_32/model.pt");
	//  module = torch::jit::load("/home/atif/neuraloperator/ico-turb/autoreg5/autoreg5_1000_8_8_100_0.5_0.001_64/model.pt");
	//}
	//catch (const c10::Error& e) {
	//  std::cerr << "error loading the model\n";
	//  return -1;
	//}
	//
	//std::cout << "ok\n";
	//
	//// Create a vector of inputs.
	//std::vector<torch::jit::IValue> inputs;
	//inputs.push_back(torch::ones({1, 10, 256, 256}));
	//
	//// Execute the model and turn its output into a tensor.
	//at::Tensor output = module.forward(inputs).toTensor();
	//std::cout << output.sizes() << " shape and slice " ;//<< output.slice(/*dim=*/1, /*start=*/0, /*end=*/1) << '\n';







        for (;;)
        {
            gettimeofday(&tv1, NULL);
	    FT_Propagate(front);
	    l_cartesian->solve(front->dt);
	    printf("Solved NS equations\n");
	    v_cartesian->recordTKE();

#ifdef __PRDNS_TIMER__
            gettimeofday(&tv3, NULL);
#endif
	    if (eqn_params->if_volume_force && front->time < 0.0001)
	    {
                v_cartesian->solve(0.0);
	    }
	    else
	    {
                 v_cartesian->solve(front->dt);
                 printf("Solved vapor and temperature equations\n");

#ifdef __PRDNS_TIMER__
                 gettimeofday(&tv4, NULL);
#endif
                 if (eqn_params->prob_type == PARTICLE_TRACKING)
                 {
                    ParticlePropagate(front);
#ifdef __CUDA__
                    v_cartesian->uploadParticle();
#endif
                 }
#ifdef __PRDNS_TIMER__
                 gettimeofday(&tv5, NULL);
                 t1 = (tv5.tv_usec - tv4.tv_usec)/1000000.0 + (tv5.tv_sec - tv4.tv_sec);
#endif
	    }
#ifdef __PRDNS_TIMER__
            gettimeofday(&tv6, NULL);
#endif

	    FT_AddTimeStepToCounter(front);
	    FT_SetTimeStep(front);
	    front->dt = FT_Min(front->dt,CFL*l_cartesian->max_dt);

            gettimeofday(&tv2, NULL);
            runtime=(tv2.tv_usec - tv1.tv_usec)/1000000.0 + (tv2.tv_sec - tv1.tv_sec);
            totaltime += runtime;
#ifdef __PRDNS_TIMER__
            printf("\n atif1 NavierStokes solver                    :  %10.2f", (tv3.tv_usec - tv1.tv_usec)/1000000.0 + (tv3.tv_sec - tv1.tv_sec));
            printf("\n atif2 Particle Propagate + Vapor temperature :  %10.2f", (tv6.tv_usec - tv3.tv_usec)/1000000.0 + (tv6.tv_sec - tv3.tv_sec));
            printf("\n atif3 Particle Propagate                     :      %10.2f", t1);
            printf("\n atif4 FT Add Set TimeStep                    :  %10.2f", (tv2.tv_usec - tv6.tv_usec)/1000000.0 + (tv2.tv_sec - tv6.tv_sec));
#endif
            printf("\nruntime = %10.2f,   total runtime = %10.2f,  time = %10.9f   step = %7d   dt = %10.9f\n\n\n",
                            runtime, totaltime, front->time,front->step,front->dt);
            fflush(stdout);
	    
            if (FT_IsSaveTime(front))
	    {
                printf("Recording data for post analysis ...\n");
		if (eqn_params->prob_type == PARTICLE_TRACKING)
		    v_cartesian->output();
	    }
            if (FT_IsMovieFrameTime(front))
	    {
		printf("Output movie frame...\n");
		// Front standard output
	 	if(movie_option->plot_particles == YES)
		{
		    vtk_plot_scatter(front);
		    vtk_plot_sample_traj(front);
		}
                FT_AddMovieFrame(front,out_name2,YES);
	    }

            if (FT_TimeLimitReached(front))
	    {
		if(movie_option->plot_particles == YES)
                    vtk_plot_scatter(front);
	    	FT_AddMovieFrame(front,out_name2,YES);
                break;
	    }
	    /* Output section, next dt may be modified */

	    FT_TimeControlFilter(front);
        }
}       /* end melting_flow_driver */

static void read_movie_options(
        char *inname,
        PARAMS *params)
{
        static MOVIE_OPTION *movie_option;
        FILE *infile = fopen(inname,"r");
        char string[100];

        FT_ScalarMemoryAlloc((POINTER*)&movie_option,sizeof(MOVIE_OPTION));
        params->movie_option = movie_option;
	movie_option->plot_temperature = NO;
	movie_option->plot_vapor = NO;
	movie_option->plot_particles = NO; /*Default option*/

        CursorAfterString(infile,"Type y to make movie of temperature:");
        fscanf(infile,"%s",string);
	(void) printf("%s\n",string);
        if (string[0] == 'Y' || string[0] == 'y')
            movie_option->plot_temperature = YES;
        CursorAfterString(infile,"Type y to make movie of vapor mixing ratio:");
        fscanf(infile,"%s",string);
	(void) printf("%s\n",string);
        if (string[0] == 'Y' || string[0] == 'y')
            movie_option->plot_vapor = YES;
	if (!params->no_droplets)
	{
	    CursorAfterString(infile,"Type y to make movie of particles:");
            fscanf(infile,"%s",string);
	    (void) printf("%s\n",string);
            if (string[0] == 'Y' || string[0] == 'y')
                movie_option->plot_particles = YES;
	}
	/* Default: not plot cross sectional variables */
        movie_option->plot_cross_section[0] = NO;
        movie_option->plot_cross_section[1] = NO;
        movie_option->plot_cross_section[2] = NO;
        fclose(infile);
}       /* end read_movie_options */

extern  double getStateTemperature(
        POINTER state)
{
        STATE *T_state = (STATE*)state;
        return T_state->temperature;
}       /* end getStateTemperature */

extern  double getStateVapor(
        POINTER state)
{
        STATE *T_state = (STATE*)state;
        return T_state->vapor;
}       /* end getStateVapor */

extern  double getStateSuper(
        POINTER state)
{
        STATE *T_state = (STATE*)state;
        return T_state->supersat;
}       /* end getStateSuper */

extern  void assignStateTemperature(
	double T,
        POINTER state)
{
        STATE *T_state = (STATE*)state;
        T_state->temperature = T;
}       /* end assignStateTemperature */

extern  void assignStateVapor(
	double T,
        POINTER state)
{
        STATE *T_state = (STATE*)state;
        T_state->vapor = T;
}       /* end assignStateVapor */
