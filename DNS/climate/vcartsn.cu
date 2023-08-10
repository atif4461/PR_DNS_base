
#include "climate.h"

//#include <cstudio>
#include <iostream>

#include <cuda_runtime.h>


using namespace std;



#ifdef __CUDA__


double* g_particle_buffer;
double* g_particle_buffer_D;
int g_max_num_particle;
double* g_particle_input;
double* g_particle_input_D;


//#define PI 3.14159265359


//*
//#if __CUDA_ARCH__ < 600
__device__ double atomicAddN(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
//#endif
//*/



void VCARTESIAN::initDevice() {
    
  cout << "[CUDA] : VCARTESIAN::initDevice() CALLED!!!" << endl;


  cudaError_t ierr;

  initFlg = 1; // initialization status;

  int count;
  int myid = pp_mynode();

  const char* env_p = std::getenv("SLURMD_NODENAME");
  cout << "[CUDA] : VCARTESIAN::initDevice() : Node Name : " << env_p << endl;

  ierr = cudaGetDeviceCount(&count);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

  cout << "[CUDA] : VCARTESIAN::initDevice() : Device Count : " << count << endl;
  cout << "[CUDA] : VCARTESIAN::initDevice() : Num Nodes : " << pp_numnodes() << endl;
  cout << "[CUDA] : VCARTESIAN::initDevice() : My ID : " << myid << endl;

  ierr = cudaSetDevice(myid % count);

  /*
  // Moved to Global
  max_num_particle = 100000000; // 100 M

  //particle_buffer = new double[max_num_particle*5]; // radius, x,y,z, rho 
  ierr = cudaMallocHost((void**)&particle_buffer, 5*max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : cudaMallocHost(particle_buffer) : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&particle_buffer_D, 5*max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : cudaMalloc(particle_buffer_D) :" << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  */

  max_comp_size = 100000000; // 100 M
  ierr = cudaMalloc((void**)&source_D, max_comp_size*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&drops_D, max_comp_size*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&cloud_D, max_comp_size*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&supersat_D, max_comp_size*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::initDevice() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

}

void VCARTESIAN::cleanDevice() {
  cudaFree(supersat_D);
  cudaFree(cloud_D);
  cudaFree(drops_D);
  cudaFree(source_D);
  //cudaFree(particle_buffer_D);
  //cudaFree(particle_buffer);
}

void VCARTESIAN::initOutput() {
  cudaMemset((void*)source_D, 0, sizeof(double)*comp_size);
  cudaMemset((void*)drops_D, 0, sizeof(double)*comp_size);
  cudaMemset((void*)cloud_D, 0, sizeof(double)*comp_size);
}

void VCARTESIAN::uploadParticle() {
  //cout << "[CUDA] : VCARTESIAN::uploadParticle() CALLED!!!" << endl;
  cudaError_t ierr;

  PARTICLE* particles = eqn_params->particle_array;
  int num_particles = eqn_params->num_drops;

  double* rho = g_particle_buffer;
  double* radius = g_particle_buffer + num_particles;
  double* x = g_particle_buffer + 2*num_particles;
  double* y = g_particle_buffer + 3*num_particles;
  double* z = g_particle_buffer + 4*num_particles;

  for(int i=0 ; i<num_particles ; i++) {
    rho[i] = particles[i].rho;
    radius[i] = particles[i].radius;
    x[i] = particles[i].center[0];
    y[i] = particles[i].center[1];
    z[i] = particles[i].center[2];
  }


  ierr = cudaMemcpy(g_particle_buffer_D, g_particle_buffer, 5*num_particles*sizeof(double), cudaMemcpyHostToDevice);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::uploadParticle() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }


}

void VCARTESIAN::uploadSupersat() {

  //cout << "[CUDA] : VCARTESIAN::uploadSupersat() CALLED!!!" << endl;

  cudaError_t ierr = cudaMemcpy(supersat_D, eqn_params->field->supersat, comp_size*sizeof(double), cudaMemcpyHostToDevice);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::uploadSupersat() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }



}



__global__ void computeDropsInCell(int num_drops, double* particles, double* supersat, double* source, double* drops, double* cloud, double topLx, double topLy, double topLz, double tophx, double tophy, double tophz, int gmax0, int gmax1, double K, double rho_a) {

  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_drops) {
    double rho = particles[i];
    double radius = particles[i + num_drops];
    double x = particles[i + 2*num_drops];
    double y = particles[i + 3*num_drops];
    double z = particles[i + 4*num_drops];

    double icx = floor((x - topLx + 0.5*tophx)/tophx);
    double icy = floor((y - topLy + 0.5*tophy)/tophy);
    double icz = floor((z - topLz + 0.5*tophz)/tophz);

    int index = (icz*(gmax1 + 1)+icy) * (gmax0 + 1) + icx;

    double src = (-4000.0*PI*rho*K/rho_a)*supersat[index]*radius;
    double cld = (4.0/3.0)*PI*pow(radius, 3.0)*rho/rho_a;

    atomicAdd(&source[index], src);
    //if(radius > 0) {
      atomicAdd(&drops[index], 1.0);
      atomicAdd(&cloud[index], cld);
    //}

  }



}


__global__ void computeTemperatureSource_Kernel(int num_drops, double* particles, double* supersat, double* source, double topLx, double topLy, double topLz, double tophx, double tophy, double tophz, int gmax0, int gmax1, double K, double rho_a, double lcp) {

  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_drops) {
    double rho = particles[i];
    double radius = particles[i + num_drops];
    double x = particles[i + 2*num_drops];
    double y = particles[i + 3*num_drops];
    double z = particles[i + 4*num_drops];

    double icx = floor((x - topLx + 0.5*tophx)/tophx);
    double icy = floor((y - topLy + 0.5*tophy)/tophy);
    double icz = floor((z - topLz + 0.5*tophz)/tophz);

    int index = (icz*(gmax1 + 1)+icy) * (gmax0 + 1) + icx;

    double src = (lcp*4.0*PI*rho*K/rho_a)*supersat[index]*radius;

    atomicAdd(&source[index], src);

  }



}





void VCARTESIAN::computeVaporSource_DropsInCell(int gmax0, int gmax1, double rho_0, double a3) {

    //cout << "[CUDA] VCARTESIAN::computeVaporSource_DropsInCell() CALLED!!!" << endl;
    int num_drops = eqn_params->num_drops;

    int threads = 1024;
    int blocks = num_drops/threads + (num_drops%threads ? 1 : 0);

    cout << "[CUDA] VCARTESIAN::computeVaporSource_DropsInCell() : blocks=" << blocks << endl;

    computeDropsInCell<<<blocks, threads>>>(num_drops, g_particle_buffer_D, supersat_D, source_D, drops_D, cloud_D, top_L[0], top_L[1], top_L[2], top_h[0], top_h[1], top_h[2], gmax0, gmax1, eqn_params->K, rho_0*a3);
    cudaDeviceSynchronize();

    cudaError_t ierr = cudaGetLastError();
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : VCARTESIAN::computeVaporSource_DropsInCell() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

}


void VCARTESIAN::retrieveResult(double* source, double* drops, double* cloud) {

  cudaError_t ierr = cudaMemcpy(source, source_D, comp_size*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::retrieveResult() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

  ierr = cudaMemcpy(drops, drops_D, comp_size*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::retrieveResult() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

  ierr = cudaMemcpy(cloud, cloud_D, comp_size*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::retrieveResult() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

}










void VCARTESIAN::computeVaporSource_CUDA(int num_drops, PARAMS* eqn_params, int gmax0, int gmax1, double rho_0, double a3, double lcp) {

    int threads = 1024;
    //int threads = 32;
    int blocks = num_drops/threads + (num_drops%threads ? 1 : 0);

    cout << "[CUDA] VCARTESIAN::computeVaporSource_CUDA() : blocks=" << blocks << endl;

    computeTemperatureSource_Kernel<<<blocks, threads>>>(num_drops, g_particle_buffer_D, supersat_D, source_D, top_L[0], top_L[1], top_L[2], top_h[0], top_h[1], top_h[2], gmax0, gmax1, eqn_params->K, rho_0*a3, lcp);
    cudaDeviceSynchronize();
    
    //cout << "[CUDA] VCARTESIAN::computeVaporSource_CUDA() : 2!!!" << endl;

    cudaError_t ierr = cudaGetLastError();
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : VCARTESIAN::computeVaporSource_CUDA() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

    //cout << "[CUDA] VCARTESIAN::computeVaporSource_CUDA() : 3!!!" << endl;


}


void VCARTESIAN::retrieveSource(double* source) {

  cudaError_t ierr = cudaMemcpy(source, source_D, comp_size*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : VCARTESIAN::retrieveResult() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
}





__global__ void propagateParticle_Kernel_c1s1(int num_drops, double* particles, double* particle_input, double rho0mu, double K, double dt, double gx, double gy, double gz) {

  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_drops) {
    double R = particles[i + num_drops];

    if(R > 0.0) {

      double rho = particles[i];
      //double x = particles[i + 2*num_drops];
      //double y = particles[i + 3*num_drops];
      //double z = particles[i + 4*num_drops];
      double u = particles[i + 5*num_drops];
      double v = particles[i + 6*num_drops];
      double w = particles[i + 7*num_drops];

      double u0 = particle_input[i];
      double v0 = particle_input[i + num_drops];
      double w0 = particle_input[i + 2*num_drops];
      double s = particle_input[i + 3*num_drops];


      double tau_p = 2*rho*R*R/(9*rho0mu);
      double exp_p = exp(-dt/tau_p);

      double delta_R = R*R + 2*K*s*dt;

      if(delta_R < 0)
        R = 0.0;
      else
        R = sqrt(delta_R);

      particles[i] = R;

      double temp = tau_p*(1 - exp_p);

      particles[i + 2*num_drops] += temp*u + (dt - temp) * (u0 + gx*tau_p);
      particles[i + 5*num_drops] = exp_p*u + (1 - exp_p) * (u0 + gx*tau_p);

      particles[i + 3*num_drops] += temp*v + (dt - temp) * (v0 + gy*tau_p);
      particles[i + 6*num_drops] = exp_p*v + (1 - exp_p) * (v0 + gy*tau_p);

      particles[i + 4*num_drops] += temp*w + (dt - temp) * (w0 + gz*tau_p);
      particles[i + 7*num_drops] = exp_p*w + (1 - exp_p) * (w0 + gz*tau_p);


    }

  }

}


__global__ void propagateParticle_Kernel_c1s1_1node(int num_drops, double* particles, double* particle_input, double rho0mu, double K, double dt, double gx, double gy, double gz, double Ux, double Lx, double Uy, double Ly, double Uz, double Lz) {

  register int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_drops) {
    double R = particles[i + num_drops];

    if(R > 0.0) {

      double rho = particles[i];
      //double x = particles[i + 2*num_drops];
      //double y = particles[i + 3*num_drops];
      //double z = particles[i + 4*num_drops];
      double u = particles[i + 5*num_drops];
      double v = particles[i + 6*num_drops];
      double w = particles[i + 7*num_drops];

      double u0 = particle_input[i];
      double v0 = particle_input[i + num_drops];
      double w0 = particle_input[i + 2*num_drops];
      double s = particle_input[i + 3*num_drops];


      double tau_p = 2*rho*R*R/(9*rho0mu);
      double exp_p = exp(-dt/tau_p);

      double delta_R = R*R + 2*K*s*dt;

      if(delta_R < 0)
        R = 0.0;
      else
        R = sqrt(delta_R);

      particles[i] = R;

      double temp = tau_p*(1 - exp_p);

      particles[i + 2*num_drops] += temp*u + (dt - temp) * (u0 + gx*tau_p);
      particles[i + 5*num_drops] = exp_p*u + (1 - exp_p) * (u0 + gx*tau_p);

      particles[i + 3*num_drops] += temp*v + (dt - temp) * (v0 + gy*tau_p);
      particles[i + 6*num_drops] = exp_p*v + (1 - exp_p) * (v0 + gy*tau_p);

      particles[i + 4*num_drops] += temp*w + (dt - temp) * (w0 + gz*tau_p);
      particles[i + 7*num_drops] = exp_p*w + (1 - exp_p) * (w0 + gz*tau_p);


      if (particles[i + 2*num_drops] > Ux)
          particles[i + 2*num_drops] = Lx + fmod(particles[i + 2*num_drops], Ux - Lx);
      if (particles[i + 2*num_drops] < Lx)
          particles[i + 2*num_drops] = Ux + fmod(particles[i + 2*num_drops], Ux - Lx);

      if (particles[i + 3*num_drops] > Uy)
          particles[i + 3*num_drops] = Ly + fmod(particles[i + 3*num_drops], Uy - Ly);
      if (particles[i + 3*num_drops] < Ly)
          particles[i + 3*num_drops] = Uy + fmod(particles[i + 3*num_drops], Uy - Ly);

      if (particles[i + 4*num_drops] > Uz)
          particles[i + 4*num_drops] = Lz + fmod(particles[i + 4*num_drops], Uz - Lz);
      if (particles[i + 4*num_drops] < Lz)
          particles[i + 4*num_drops] = Uz + fmod(particles[i + 4*num_drops], Uz - Lz);




    }

  }

}










void ParticlePropagate_CUDA(int num_drops, bool condensation, bool sedimentation, double rho0mu, double K, double dt, double gx, double gy, double gz) {
    cout << "[CUDA] : ParticlePropagate_CUDA() CALLED!!!" << endl;


    int threads = 1024;
    int blocks = num_drops/threads + (num_drops%threads ? 1 : 0);

    cudaError_t ierr = cudaMemcpy(g_particle_input_D, g_particle_input, 4*num_drops*sizeof(double), cudaMemcpyHostToDevice);
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : ParticlePropagate_CUDA() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

    propagateParticle_Kernel_c1s1<<<blocks, threads>>>(num_drops, g_particle_buffer_D, g_particle_input_D, rho0mu, K, dt, gx, gy, gz);
    cudaDeviceSynchronize();
    ierr = cudaGetLastError();
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : ParticlePropagate_CUDA() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

}


void ParticlePropagate_CUDA_1node(int num_drops, bool condensation, bool sedimentation, double rho0mu, double K, double dt, double gx, double gy, double gz, double Ux, double Lx, double Uy, double Ly, double Uz, double Lz) {
    cout << "[CUDA] : ParticlePropagate_CUDA() CALLED!!!" << endl;


    int threads = 1024;
    int blocks = num_drops/threads + (num_drops%threads ? 1 : 0);

    cudaError_t ierr = cudaMemcpy(g_particle_input_D, g_particle_input, 4*num_drops*sizeof(double), cudaMemcpyHostToDevice);
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : ParticlePropagate_CUDA() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

    propagateParticle_Kernel_c1s1_1node<<<blocks, threads>>>(num_drops, g_particle_buffer_D, g_particle_input_D, rho0mu, K, dt, gx, gy, gz, Ux, Lx, Uy, Ly, Uz, Lz);
    cudaDeviceSynchronize();
    ierr = cudaGetLastError();
    if(cudaSuccess != ierr) {
      cout << "[CUDA] : ParticlePropagate_CUDA() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
    }

}




void initDeviceParticle() {
  cout << "[CUDA] : initDeviceParticle() CALLED!!!" << endl;


  cudaError_t ierr;


  int count;
  int myid = pp_mynode();

  ierr = cudaGetDeviceCount(&count);
  cout << "[CUDA] : initDeviceParticle() : Device Count : " << count << endl;
  cout << "[CUDA] : initDeviceParticle() : Num Nodes : " << pp_numnodes() << endl;
  cout << "[CUDA] : initDeviceParticle() : My ID : " << myid << endl;

  ierr = cudaSetDevice(myid);


  g_max_num_particle = 100000000; // 100 M

  /*
   * particle_buffer_D structure
   * rho, radius, x, y, z, u, v, w
   *
   */

  //particle_buffer = new double[max_num_particle*5]; // rho, radius, x,y,z, u,v,w
  ierr = cudaMallocHost((void**)&g_particle_buffer, 8*g_max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : initDeviceParticle() Error!!! : cudaMallocHost(g_particle_buffer) : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&g_particle_buffer_D, 8*g_max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : initDeviceParticle() Error!!! : cudaMalloc(g_particle_buffer_D) :" << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

  ierr = cudaMallocHost((void**)&g_particle_input, 4*g_max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : initDeviceParticle() Error!!! : cudaMallocHost(g_particle_input) : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }
  ierr = cudaMalloc((void**)&g_particle_input_D, 4*g_max_num_particle*sizeof(double));
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : initDeviceParticle() Error!!! : cudaMalloc(g_particle_input_D) :" << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }

}

void clearDeviceParticle() {
  cudaFree(g_particle_buffer_D);
  cudaFree(g_particle_buffer);
  cudaFree(g_particle_input_D);
  cudaFree(g_particle_input);
}

void uploadParticle(int num_drops, PARTICLE* particles) {

  cudaError_t ierr;

  double* rho = g_particle_buffer;
  double* radius = g_particle_buffer + num_drops;
  double* x = g_particle_buffer + 2*num_drops;
  double* y = g_particle_buffer + 3*num_drops;
  double* z = g_particle_buffer + 4*num_drops;
  double* u = g_particle_buffer + 5*num_drops;
  double* v = g_particle_buffer + 6*num_drops;
  double* w = g_particle_buffer + 7*num_drops;

  for(int i=0 ; i<num_drops ; i++) {
    rho[i] = particles[i].rho;
    radius[i] = particles[i].radius;
    x[i] = particles[i].center[0];
    y[i] = particles[i].center[1];
    z[i] = particles[i].center[2];
    u[i] = particles[i].vel[0];
    v[i] = particles[i].vel[1];
    w[i] = particles[i].vel[2];
  }


  ierr = cudaMemcpy(g_particle_buffer_D, g_particle_buffer, 8*num_drops*sizeof(double), cudaMemcpyHostToDevice);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : uploadParticle() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }



}

void downloadParticle(int num_drops, PARTICLE* particles) {

  cudaError_t ierr = cudaMemcpy(g_particle_buffer + num_drops, g_particle_buffer_D + num_drops, 7*num_drops*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaSuccess != ierr) {
    cout << "[CUDA] : downloadParticle()() Error!!! : " << ierr << ", " << cudaGetErrorString(ierr) << endl;
  }


  double* radius = g_particle_buffer + num_drops;
  double* x = g_particle_buffer + 2*num_drops;
  double* y = g_particle_buffer + 3*num_drops;
  double* z = g_particle_buffer + 4*num_drops;
  double* u = g_particle_buffer + 5*num_drops;
  double* v = g_particle_buffer + 6*num_drops;
  double* w = g_particle_buffer + 7*num_drops;

  for(int i=0 ; i<num_drops ; i++) {
    particles[i].radius = radius[i];
    particles[i].center[0] = x[i];
    particles[i].center[1] = y[i];
    particles[i].center[2] = z[i];
    particles[i].vel[0] = u[i];
    particles[i].vel[1] = v[i];
    particles[i].vel[2] = w[i];
  }

}





#endif




