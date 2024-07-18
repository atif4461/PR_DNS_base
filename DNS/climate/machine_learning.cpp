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
void VCARTESIAN::transformVel2Dprdns2torch (
    int time, at::Tensor &input){

    at::Tensor vel_x = torch::zeros({256, 256}, torch::dtype(torch::kFloat32));
    at::Tensor vel_y = torch::zeros({256, 256}, torch::dtype(torch::kFloat32));
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
	int channel = time;
	vel_x[row][col] = static_cast<float>(vec[0]); // ux
	vel_y[row][col] = static_cast<float>(vec[1]); // uy
	input[0][channel] = torch::cat({vel_x, vel_y}, 0);
    }


}

/**
 * \brief Loads a velocity field snapshot in at::Tensor.
 * 
 * For 2d fields, this transforms matrix to Cartesian 
 * notation
 *
 * (0,0)            (0,255)          (0,255)        (255,255)    
 *     _________________               _________________         
 *    |                |              |                |         
 *    |                |              |                |         
 *    .                .       -----> .                .         
 *    |                |              |                |         
 *    |________________|              |________________|         
 * (255,0)          (255,255)        (0,0)          (255,0)      
 *
 * \return void.
 */
void VCARTESIAN::transformVel2Dtorch2prdns (
    at::Tensor &tensor){

    int64_t split_size = 256;
    auto tensors = torch::split(tensor, split_size, 0);

    std::cout << tensors[0].sizes() << " split shape and slice " << tensors[1].sizes() << " " << tensors[0][0][0] << std::endl ;
    
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
    
    int index1, index2, index3, index4;

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

        vel[0][index] = tensors[0][row][col].item<double>();
        vel[1][index] = tensors[1][row][col].item<double>();

	if (i == imin)
		if ( j == jmin ) index1 = index;
		if ( j == jmax ) index2 = index;
	if (i == imax)
		if ( j == jmin ) index3 = index;
		if ( j == jmax ) index4 = index;

        //std::cout << "icords are" << i_local << " " << j_local << " " << vec[0] << "," << vec[1] << std::endl;
    }

    std::cout << "TENSOR " << tensors[0][0][255].item<double>() << " " << tensors[0][255][255].item<double>() << " " << tensors[0][255][0].item<double>() << " " << tensors[0][0][0].item<double>() << std::endl;
    std::cout << "PRDNS  " << vel[0][index1] << " " << vel[0][index2] << " " << vel[0][index3] << " " << vel[0][index4] << std::endl;

    //// TODO: Copy halos here
    for (l = 0; l < dim; ++l)
    {
        FT_ParallelExchGridArrayBuffer(vel[l],front,NULL);
    }


}


