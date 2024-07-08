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

/**
 * \brief Loads a velocity field snapshot in at::Tensor.
 * 
 * For 2d fields, this transforms Cartesian to matrix 
 * notation
 *
 * (0,255)        (255,255)           (0,0)            (0,255)
 *   _________________                    _________________
 *  |                |                   |                |
 *  |                |                   |                |
 *  .                .       ----->      .                .
 *  |                |                   |                |
 *  |________________|                   |________________|    
 * (0,0)          (255,0)             (255,0)          (255,255)
 *
 * \return void.
 */
void VCARTESIAN::transformVel2Dprdns2Torch (
    int time, at::Tensor &input){

    std::cout << "Saving " << time << " in tensor " << std::endl;

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

        int i_local = i - imin;
        int j_local = j - jmin;

        int row = 255 - j_local;
	int col = i_local;

        icoords[0] = i;
        icoords[1] = j;
        index  = d_index(icoords,gmax,dim);

        for (l = 0; l < dim; ++l)
    	   vec[l] = vel[l][index];

	//std::cout << "icords are" << i_local << " " << j_local << " " << vec[0] << "," << vec[1] << std::endl;
	int channel = 2*time;
	input[0][channel  ][row][col] = static_cast<float>(vec[0]); // ux
	input[0][channel+1][row][col] = static_cast<float>(vec[1]); // uy
    }


}


