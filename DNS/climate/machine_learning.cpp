/********************************************************************
 * 		machine_learning.cpp
 * This file contains the functions related to ML inferences in PRDNS
 * Author: Atif, CSI, BNL
 *******************************************************************/

#include <iFluid.h>
#include "solver.h"
#include "climate.h"
#include <torch/torch.h>
#include <torch/script.h>

void VCARTESIAN::transformVelPrdns2Torch (
    int time,
    std::vector<torch::jit::IValue> &inputs
    ){

    INTERFACE *grid_intfc = front->grid_intfc;
    RECT_GRID *gr = &topological_grid(grid_intfc);
    int gmax[MAXD],icoords[MAXD];
    int i,j,k,l,index;
    int dim = grid_intfc->dim;
    double vec[MAXD];
    double **vel = field->vel;
    
    for (i = 0; i < dim; ++i){
        gmax[i] = gr->gmax[i]; 
    }
    
    for (l = 0; l < MAXD; ++l) 
	vec[l] = 0.0;
    
    for (j = jmin; j <= jmax; j++)
    for (i = imin; i <= imax; i++){
    // The below loop gives access to halos
    //for (j = 0; j <= gmax[1]; j++)
    //for (i = 0; i <= gmax[0]; i++){	

        icoords[0] = i;
        icoords[1] = j;
        index  = d_index(icoords,gmax,dim);
        for (l = 0; l < dim; ++l)
    	   vec[l] = vel[l][index];
	
	std::cout << "iicoords are" << i << " " << j << std::endl;
        //fprintf(vfile,"%f %f %f\n",vec[0],vec[1],vec[2]);
    }



}

