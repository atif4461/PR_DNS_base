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


#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
/*
*	Perform a n-dimensional FFT
*	For real to complex transform, use dir = 1: 
	    input:  N0xN1x...xN{n-1} real array
	    output: N0xN1x...xN{n-1}/2+1 complex array
	For complex to real transform, use dir = -1:
	    input:  N0xN1x...xN{n-1}/2+1 complex array
	    output:  N0xN1x...xN{n-1} real array
*
*/
extern bool fftnd(
	fftw_complex *in,
	int dim, /*rank of array, can be 1, 2, 3 or any positive integer*/
	int *N,  /*number in each dim*/
	int dir)
{
	fftw_complex *out;
	fftw_plan p;
	int Nc, Nr, i;
	unsigned int flags = FFTW_ESTIMATE; /*or FFTW_MEASURE*/
	static fftw_complex *cplx_array;
	static double *real_array;
	static int N_max = 0;

	/*count total number for input and output*/
	Nc = 1;
	for (i = 0; i < dim-1; i++)
	    Nc *= N[i];
	Nc *= N[dim-1]/2 + 1; /*last dim is cut to (N/2)+1 for complex array*/
	Nr = 1;
	for (i = 0; i < dim; i++)
	    Nr *= N[i];	      /*no cut for real array*/
	
	if (Nr > N_max)
	{
	    N_max = Nr;
            fftw_free(cplx_array);
            fftw_free(real_array);
	    real_array = new double[Nr];
	    cplx_array = new fftw_complex[Nc];
	}

	switch (dir)
	{
	    case 1:
		p = fftw_plan_dft_r2c(dim,N,real_array,cplx_array,flags);
		for (i = 0; i < Nr; i++ )
		    real_array[i] = in[i][0];
		break;
	    case -1:
		p = fftw_plan_dft_c2r(dim,N,cplx_array,real_array,flags);
		for (i = 0; i < Nc; i++ )
		{
		    cplx_array[i][0] = in[i][0]; /*normalization*/
		    cplx_array[i][1] = in[i][1]; /*normalization*/
		    //printf("ifftw3 %d %f %f\n", i,cplx_array[i][0],cplx_array[i][1]);
		}
		break;
	    default:
		printf("Dir can only be -1 and 1 in FFT: unknown %d\n",dir);
		break;  
	}	

	fftw_execute(p); /*excute FFT*/

	switch(dir)
	{
	    case 1:
		for (i = 0; i < Nc; i++)
		{
		    in[i][0] = cplx_array[i][0]/Nr;
		    in[i][1] = cplx_array[i][1]/Nr;	
		}
		break;
	    case -1:
		for (i = 0; i < Nr; i++)
		{		
    		    in[i][0] = real_array[i];
		    in[i][1] = 0.0;
		}
		break;
	    default:
                printf("Dir can only be -1 and 1 in FFT: unknown %d\n",dir);
                break;
	}
	/*destroy plan*/
	fftw_destroy_plan(p);
}

/*following functions are for verification purpose*/
void printMatrix(const char*,fftw_complex*,int);

int one_dim_test()
{
	const static int N = 100;
	fftw_complex mycomplex[N];
	int dim[1];
	dim[0] = N;
	
	FILE *file;
	int i;

	for (i = 0; i < N; i++)
	{
	    mycomplex[i][0] = i;//0.7*sin(2*M_PI*50*i/(N-1))+sin(2*M_PI*120*i/(N-1));
	    mycomplex[i][1] = 0.0;
	}

	printMatrix("fft1d_input",mycomplex,N);
	fftnd(mycomplex,1,dim,1);
	printMatrix("fft1d_fft",mycomplex,N);
	fftnd(mycomplex,1,dim,-1);
	//printMatrix("fft1d_ifft",mycomplex,N);
	fftnd(mycomplex,1,dim,-1);
	printMatrix("fft1d_ifft",mycomplex,N);
	return 1;
}

int one_dim_ifft_test()
{
	const static int N = 100;
	fftw_complex mycomplex[N];
	int dim[1];
	dim[0] = N;
	
	FILE *file;
	int i;

	for (i = 0; i < N; i++)
	{
	    mycomplex[i][0] = i;
	    mycomplex[i][1] = 0.0;
	}

	printMatrix("fft1d_input",mycomplex,N);
	fftnd(mycomplex,1,dim,-1);
	printMatrix("fft1d_ifft",mycomplex,N);

	fftw_complex fftcomplex[N];

        fftcomplex[0 ][0] = 49.500000; fftcomplex[0 ][1] = 0.000000; 
        fftcomplex[1 ][0] = -0.500000; fftcomplex[1 ][1] = 15.910258;
        fftcomplex[2 ][0] = -0.500000; fftcomplex[2 ][1] = 7.947272;
        fftcomplex[3 ][0] = -0.500000; fftcomplex[3 ][1] = 5.289447;
        fftcomplex[4 ][0] = -0.500000; fftcomplex[4 ][1] = 3.957908;
        fftcomplex[5 ][0] = -0.500000; fftcomplex[5 ][1] = 3.156876;
        fftcomplex[6 ][0] = -0.500000; fftcomplex[6 ][1] = 2.621092;
        fftcomplex[7 ][0] = -0.500000; fftcomplex[7 ][1] = 2.236871;
        fftcomplex[8 ][0] = -0.500000; fftcomplex[8 ][1] = 1.947371;
        fftcomplex[9 ][0] = -0.500000; fftcomplex[9 ][1] = 1.721011;
        fftcomplex[10][0] = -0.500000; fftcomplex[10][1] = 1.538842;
        fftcomplex[11][0] = -0.500000; fftcomplex[11][1] = 1.388803;
        fftcomplex[12][0] = -0.500000; fftcomplex[12][1] = 1.262856;
        fftcomplex[13][0] = -0.500000; fftcomplex[13][1] = 1.155432;
        fftcomplex[14][0] = -0.500000; fftcomplex[14][1] = 1.062554;
        fftcomplex[15][0] = -0.500000; fftcomplex[15][1] = 0.981305;
        fftcomplex[16][0] = -0.500000; fftcomplex[16][1] = 0.909497;
        fftcomplex[17][0] = -0.500000; fftcomplex[17][1] = 0.845454;
        fftcomplex[18][0] = -0.500000; fftcomplex[18][1] = 0.787874;
        fftcomplex[19][0] = -0.500000; fftcomplex[19][1] = 0.735728;
        fftcomplex[20][0] = -0.500000; fftcomplex[20][1] = 0.688191;
        fftcomplex[21][0] = -0.500000; fftcomplex[21][1] = 0.644596;
        fftcomplex[22][0] = -0.500000; fftcomplex[22][1] = 0.604396;
        fftcomplex[23][0] = -0.500000; fftcomplex[23][1] = 0.567139;
        fftcomplex[24][0] = -0.500000; fftcomplex[24][1] = 0.532446;
        fftcomplex[25][0] = -0.500000; fftcomplex[25][1] = 0.500000;
        fftcomplex[26][0] = -0.500000; fftcomplex[26][1] = 0.469531;
        fftcomplex[27][0] = -0.500000; fftcomplex[27][1] = 0.440809;
        fftcomplex[28][0] = -0.500000; fftcomplex[28][1] = 0.413636;
        fftcomplex[29][0] = -0.500000; fftcomplex[29][1] = 0.387840;
        fftcomplex[30][0] = -0.500000; fftcomplex[30][1] = 0.363271;
        fftcomplex[31][0] = -0.500000; fftcomplex[31][1] = 0.339800;
        fftcomplex[32][0] = -0.500000; fftcomplex[32][1] = 0.317310;
        fftcomplex[33][0] = -0.500000; fftcomplex[33][1] = 0.295699;
        fftcomplex[34][0] = -0.500000; fftcomplex[34][1] = 0.274877;
        fftcomplex[35][0] = -0.500000; fftcomplex[35][1] = 0.254763;
        fftcomplex[36][0] = -0.500000; fftcomplex[36][1] = 0.235282;
        fftcomplex[37][0] = -0.500000; fftcomplex[37][1] = 0.216369;
        fftcomplex[38][0] = -0.500000; fftcomplex[38][1] = 0.197964;
        fftcomplex[39][0] = -0.500000; fftcomplex[39][1] = 0.180011;
        fftcomplex[40][0] = -0.500000; fftcomplex[40][1] = 0.162460;
        fftcomplex[41][0] = -0.500000; fftcomplex[41][1] = 0.145263;
        fftcomplex[42][0] = -0.500000; fftcomplex[42][1] = 0.128378;
        fftcomplex[43][0] = -0.500000; fftcomplex[43][1] = 0.111763;
        fftcomplex[44][0] = -0.500000; fftcomplex[44][1] = 0.095380;
        fftcomplex[45][0] = -0.500000; fftcomplex[45][1] = 0.079192;
        fftcomplex[46][0] = -0.500000; fftcomplex[46][1] = 0.063165;
        fftcomplex[47][0] = -0.500000; fftcomplex[47][1] = 0.047264;
        fftcomplex[48][0] = -0.500000; fftcomplex[48][1] = 0.031457;
        fftcomplex[49][0] = -0.500000; fftcomplex[49][1] = 0.015713;
        fftcomplex[50][0] = -0.500000; fftcomplex[50][1] = 0.0;
	printMatrix("fft1d_nice_ifft",fftcomplex,N);
	fftnd(fftcomplex,1,dim,-1);
	printMatrix("fft1d_nice_fft",fftcomplex,N);
	return 1;
}

int two_dim_test()
{
	const static int M = 48, N = 64;
	int i,j,index;
	int dim[2];
	dim[0] = M; dim[1] = N;
	FILE *file;
	double wn, L = 1.0;

	fftw_complex myarray[M*N];

	for (j = 0; j < N; j++)
	for (i = 0; i < M; i++)
	{
	    wn = (2*M_PI/L)*sqrt(i*i+j*j);
	    index = j * M + i;
	    myarray[index][0] = index;//sin(2*M_PI*i/M)+cos(2*M_PI*j/N); 
	    myarray[index][1] = 0.0;
	}
	printMatrix("fft2d_input",myarray,M*N);
	fftnd(myarray,2,dim,1);
	printMatrix("fft2d_fft",myarray,M*N);
	fftnd(myarray,2,dim,-1);
	printMatrix("fft2d_ifft",myarray,M*N);
        return 1;
}

int three_dim_test()
{
	const static int Nx = 8, Ny = 16, Nz = 32;
	int i,j,k,index;
	FILE *file;
	double wn, L = 1.0;
	int dim[3];
	dim[0] = Nx; dim[1] = Ny; dim[2] = Nz;

	fftw_complex myarray[Nx*Ny*Nz];

	for (i = 0; i < Nx; i++)
	for (j = 0; j < Ny; j++)
	for (k = 0; k < Nz; k++)
	{
	    wn = (2*M_PI/L)*sqrt(i*i+j*j+k*k);
	    index = Nx*(Ny * k + j) + i;
	    myarray[index][0] = index;//cos(2*M_PI*i/Nx)*cos(2*M_PI*j/Ny)*sin(2*M_PI*k/Nz); 
	    myarray[index][1] = 0.0;
	}
	printMatrix("fft3d_input",myarray,Nx*Ny*Nz);
	fftnd(myarray,3,dim,1);
	printMatrix("fft3d_fft",myarray,Nx*Ny*Nz);
	fftnd(myarray,3,dim,-1);
	printMatrix("fft3d_ifft",myarray,Nx*Ny*Nz);
        return 1;
}

int three_dim_ifft_test()
{
	//const static int Nx = 8, Ny = 1, Nz = 1;//1
	const static int Nx = 8, Ny = 4, Nz = 4;
	int i,j,k,index;
	FILE *file;
	double wn, L = 1.0;
	int dim[3];
	dim[0] = Nx; dim[1] = Ny; dim[2] = Nz;

	fftw_complex myarray[Nx*Ny*Nz];
	//for (i = 0; i < Nx; i++)
	//for (j = 0; j < Ny; j++)
	//for (k = 0; k < Nz; k++)
        for (k = 3; k < Nx*Ny*Nz; k++) 	
	{
	    index = Nx*(Ny * k + j) + i;
	    myarray[k][0] = 0*index;//cos(2*M_PI*i/Nx)*cos(2*M_PI*j/Ny)*sin(2*M_PI*k/Nz); 
	    myarray[k][1] = 0.0;
	}
	myarray[0][0] = 64;
        myarray[0][1] = 0;
        myarray[1][0] = 32;
        myarray[1][1] = 0;
        //myarray[2][0] = 16;
        //myarray[2][1] = 0;
        //myarray[3][0] = 8;
        //myarray[3][1] = 0;

	printMatrix("fft3d_input",myarray,Nx*Ny*Nz);
	fftnd(myarray,3,dim,-1);
	printMatrix("fft3d_ifft",myarray,Nx*Ny*Nz);
        return 1;
}



int two_dim_filter()
{
	const static int M = 64, N = 64;
	int i,j,index;
	FILE *file;
	double wn, phi, L = 1.0;

	fftw_complex myarray[M*N];
	int dim[2];
	dim [0] = M; dim[1] = N;

#if !defined __NO_RND__
	srand(time(NULL));
#endif

	for (i = 0; i < M; i++)
	for (j = 0; j < N/2+1; j++)
	{
	    index = i * (N/2+1) + j;
	    if (i * i + j *j > 4)
	    {
		myarray[index][0] = myarray[index][1] = 0.0;
		continue;
	    }
	    wn = (2*M_PI/L)*sqrt(i*i+j*j);
#if defined __NO_RND__
	    phi = 0.5;
#else		    
    	    phi  = (double)rand() / (RAND_MAX + 1.0);
#endif
	    myarray[index][0] = wn*wn*exp(-wn*wn/pow(2*M_PI*4.7568/L,2))
				 * cos(2*M_PI*phi); 
	    myarray[index][1] = wn*wn*exp(-wn*wn/pow(2*M_PI*4.7568/L,2))
				 * sin(2*M_PI*phi);
	}
	fftnd(myarray,2,dim,-1);
	printMatrix("fft2d_filter_test",myarray,M*N);
        return 1;
}

void printMatrix(const char* filename,fftw_complex* complex_array,int size)
{
	FILE* file;
	int i;
	file = fopen(filename,"w");
	for (i = 0; i < size; i++)
	{
	   fprintf(file,"%f %f\n",complex_array[i][0],complex_array[i][1]);
	}
	fclose(file);
}

