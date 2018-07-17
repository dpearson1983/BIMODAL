/* bispectrum_model.h
 * David W. Pearson
 * 17 July 2018
 * 
 * This header file will store the GPU functions needed to compute the bispectrum model. The intent
 * is to split the current bkmcmc.h header file into two part, one that calculates the model and one
 * that runs the MCMC chain.
 */

#ifndef _BISPECTRUM_MODEL_H_
#define _BISPECTRUM_MODEL_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cuda.h>
#include <vector_types.h>
#include "gpuerrchk.h"

#define TWOSEVENTHS 0.285714285714
#define THREESEVENTHS 0.428571428571
#define FOURSEVENTHS 0.571428571429
#define FIVESEVENTHS 0.714285714286
#define PI 3.1415926536

__constant__ float4 d_Pk[128]; //   2048 bytes
__constant__ float d_wi[32];   //    128 bytes
__constant__ float d_xi[32];   //    128 bytes
__constant__ float d_p[7];     //     20 bytes
// Total constant memory:           2324 out of 65536 bytes

const float w_i[] = {0.096540088514728, 0.096540088514728, 0.095638720079275, 0.095638720079275,
                     0.093844399080805, 0.093844399080805, 0.091173878695764, 0.091173878695764,
                     0.087652093004404, 0.087652093004404, 0.083311924226947, 0.083311924226947,
                     0.078193895787070, 0.078193895787070, 0.072345794108849, 0.072345794108849,
                     0.065822222776362, 0.065822222776362, 0.058684093478536, 0.058684093478536,
                     0.050998059262376, 0.050998059262376, 0.042835898022227, 0.042835898022227,
                     0.034273862913021, 0.034273862913021, 0.025392065309262, 0.025392065309262,
                     0.016274394730906, 0.016274394730906, 0.007018610009470, 0.007018610009470};

const float x_i[] = {-0.048307665687738, 0.048307665687738, -0.144471961582796, 0.144471961582796,
                     -0.239287362252137, 0.239287362252137, -0.331868602282128, 0.331868602282128,
                     -0.421351276130635, 0.421351276130635, -0.506899908932229, 0.506899908932229,
                     -0.587715757240762, 0.587715757240762, -0.663044266930215, 0.663044266930215,
                     -0.732182118740290, 0.732182118740290, -0.794483795967942, 0.794483795967942,
                     -0.849367613732570, 0.849367613732570, -0.896321155766052, 0.896321155766052,
                     -0.934906075937739, 0.934906075937739, -0.964762255587506, 0.964762255587506,
                     -0.985611511545268, 0.985611511545268, -0.997263861849481, 0.997263861849481};
                     
// Evaluates the spline to get the power spectrum at k
__device__ float pk_spline_eval(float k); // done

// Calculates a single element of the sum to get the bispectrum for a particular k triplet
__device__ double bispec_mono(int x, float &phi, float3 k); // done

// Calculates a single element of the sum to get the bispectrum for a particular k triplet
__device__ double bispec_quad(int x, float &phi, float3 k); // done

// Enables the above bispec_model to be executed on many CUDA cores to speed up the integral calculation
__global__ void bispec_gauss_32(float3 *ks, double *Bk);

__device__ float pk_spline_eval(float k) {
    int i = (k - d_Pk[0].x)/(d_Pk[1].x - d_Pk[0].x);
    
    float Pk = (d_Pk[i + 1].z*(k - d_Pk[i].x)*(k - d_Pk[i].x)*(k - d_Pk[i].x))/(6.0*d_Pk[i].w)
              + (d_Pk[i].z*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k)*(d_Pk[i + 1].x - k))/(6.0*d_Pk[i].w)
              + (d_Pk[i + 1].y/d_Pk[i].w - (d_Pk[i + 1].z*d_Pk[i].w)/6.0)*(k - d_Pk[i].x)
              + (d_Pk[i].y/d_Pk[i].w - (d_Pk[i].w*d_Pk[i].z)/6.0)*(d_Pk[i + 1].x - k);
              
    return Pk;
}

__device__ double bispec_mono(int x, float &phi, float3 k) {
    // Calculate the mu's without the AP effects
    float z = (k.x*k.x + k.y*k.y - k.z*k.z)/(2.0*k.x*k.y);
    float mu1 = d_xi[x];
    float mu2 = -d_xi[x]*z + sqrtf(1.0 - d_xi[x]*d_xi[x])*sqrtf(1.0 - z*z)*cos(phi);
    float mu3 = -(mu1*k.x + mu2*k.y)/k.z;
    
    // It's convenient to store these quantities to reduce the number of FLOP's needed later
    float sq_ratio = (d_p[4]*d_p[4])/(d_p[3]*d_p[3]) - 1.0;
    float mu1bar = 1.0 + mu1*mu1*sq_ratio;
    float mu2bar = 1.0 + mu2*mu2*sq_ratio;
    float mu3bar = 1.0 + mu3*mu3*sq_ratio;
    
    // Convert the k's and mu's to include the AP effects
    float k1 = (k.x*sqrtf(mu1bar)/d_p[4]);
    float k2 = (k.y*sqrtf(mu2bar)/d_p[4]);
    float k3 = (k.z*sqrtf(mu3bar)/d_p[4]);
    
    float P1 = pk_spline_eval(k1)/(d_p[4]*d_p[4]*d_p[3]);
    float P2 = pk_spline_eval(k2)/(d_p[4]*d_p[4]*d_p[3]);
    float P3 = pk_spline_eval(k3)/(d_p[4]*d_p[4]*d_p[3]);
    
    mu1 = (mu1*d_p[4])/(d_p[3]*sqrt(mu1bar));
    mu2 = (mu2*d_p[4])/(d_p[3]*sqrt(mu2bar));
    mu3 = (mu3*d_p[4])/(d_p[3]*sqrt(mu3bar));
    
    // More convenient things to calculate before the long expressions
    float mu12 = -(k1*k1 + k2*k2 - k3*k3)/(2.0*k1*k2);
    float mu23 = -(k2*k2 + k3*k3 - k1*k1)/(2.0*k2*k3);
    float mu31 = -(k3*k3 + k1*k1 - k2*k2)/(2.0*k3*k1);
    
    float k12 = sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12);
    float k23 = sqrt(k2*k2 + k3*k3 + 2.0*k2*k3*mu23);
    float k31 = sqrt(k3*k3 + k1*k1 + 2.0*k3*k1*mu31);
    
    float mu12p = (k1*mu1 + k2*mu2)/k12;
    float mu23p = (k2*mu2 + k3*mu3)/k23;
    float mu31p = (k3*mu3 + k1*mu1)/k31;
    
    float Z1k1 = (d_p[0] + d_p[2]*mu1*mu1);
    float Z1k2 = (d_p[0] + d_p[2]*mu2*mu2);
    float Z1k3 = (d_p[0] + d_p[2]*mu3*mu3);
    
    float F12 = FIVESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + TWOSEVENTHS*mu12*mu12;
    float F23 = FIVESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + TWOSEVENTHS*mu23*mu23;
    float F31 = FIVESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + TWOSEVENTHS*mu31*mu31;
    
    float G12 = THREESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + FOURSEVENTHS*mu12*mu12;
    float G23 = THREESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + FOURSEVENTHS*mu23*mu23;
    float G31 = THREESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + FOURSEVENTHS*mu31*mu31;
    
    float Z2k12 = 0.5*d_p[1] + d_p[0]*(F12 + 0.5*mu12p*k12*(mu1/k1 + mu2/k2)) + d_p[2]*mu12p*mu12p*G12
                  + 0.5*d_p[2]*d_p[2]*mu12p*k12*mu1*mu2*(mu1/k1 + mu2/k2) 
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu12*mu12 - 1.0/3.0);
    float Z2k23 = 0.5*d_p[1] + d_p[0]*(F23 + 0.5*mu23p*k23*(mu2/k2 + mu3/k3)) + d_p[2]*mu23p*mu23p*G23
                  + 0.5*d_p[2]*d_p[2]*mu23p*k23*mu2*mu3*(mu2/k2 + mu3/k3)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu23*mu23 - 1.0/3.0);
    float Z2k31 = 0.5*d_p[1] + d_p[0]*(F31 + 0.5*mu31p*k31*(mu3/k3 + mu1/k1)) + d_p[2]*mu31p*mu31p*G31
                  + 0.5*d_p[2]*d_p[2]*mu31p*k31*mu3*mu1*(mu3/k3 + mu1/k1)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu31*mu31 - 1.0/3.0);
                  
    float den = 1.0 + 0.5*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*d_p[5]*d_p[5];
    float FoG = 1.0/(den*den);
    
    return 2.0*(Z2k12*Z1k1*Z1k2*P1*P2 + Z2k23*Z1k2*Z1k3*P2*P3 + Z2k31*Z1k3*Z1k1*P3*P1)*FoG;
}

__device__ float legendre_quad(float mu) {
    return 0.5*(3.0*mu*mu - 1.0);
}

__device__ double bispec_quad(int x, float &phi, float3 k) {
    // Calculate the mu's without the AP effects
    float z = (k.x*k.x + k.y*k.y - k.z*k.z)/(2.0*k.x*k.y);
    float mu1 = d_xi[x];
    float mu2 = -d_xi[x]*z + sqrtf(1.0 - d_xi[x]*d_xi[x])*sqrtf(1.0 - z*z)*cos(phi);
    float mu3 = -(mu1*k.x + mu2*k.y)/k.z;
    
    float P_L = legendre_quad(mu1);
    
    // It's convenient to store these quantities to reduce the number of FLOP's needed later
    float sq_ratio = (d_p[4]*d_p[4])/(d_p[3]*d_p[3]) - 1.0;
    float mu1bar = 1.0 + mu1*mu1*sq_ratio;
    float mu2bar = 1.0 + mu2*mu2*sq_ratio;
    float mu3bar = 1.0 + mu3*mu3*sq_ratio;
    
    // Convert the k's and mu's to include the AP effects
    float k1 = (k.x*sqrtf(mu1bar)/d_p[4]);
    float k2 = (k.y*sqrtf(mu2bar)/d_p[4]);
    float k3 = (k.z*sqrtf(mu3bar)/d_p[4]);
    
    float P1 = pk_spline_eval(k1)/(d_p[4]*d_p[4]*d_p[3]);
    float P2 = pk_spline_eval(k2)/(d_p[4]*d_p[4]*d_p[3]);
    float P3 = pk_spline_eval(k3)/(d_p[4]*d_p[4]*d_p[3]);
    
    mu1 = (mu1*d_p[4])/(d_p[3]*sqrt(mu1bar));
    mu2 = (mu2*d_p[4])/(d_p[3]*sqrt(mu2bar));
    mu3 = (mu3*d_p[4])/(d_p[3]*sqrt(mu3bar));
    
    // More convenient things to calculate before the long expressions
    float mu12 = -(k1*k1 + k2*k2 - k3*k3)/(2.0*k1*k2);
    float mu23 = -(k2*k2 + k3*k3 - k1*k1)/(2.0*k2*k3);
    float mu31 = -(k3*k3 + k1*k1 - k2*k2)/(2.0*k3*k1);
    
    float k12 = sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12);
    float k23 = sqrt(k2*k2 + k3*k3 + 2.0*k2*k3*mu23);
    float k31 = sqrt(k3*k3 + k1*k1 + 2.0*k3*k1*mu31);
    
    float mu12p = (k1*mu1 + k2*mu2)/k12;
    float mu23p = (k2*mu2 + k3*mu3)/k23;
    float mu31p = (k3*mu3 + k1*mu1)/k31;
    
    float Z1k1 = (d_p[0] + d_p[2]*mu1*mu1);
    float Z1k2 = (d_p[0] + d_p[2]*mu2*mu2);
    float Z1k3 = (d_p[0] + d_p[2]*mu3*mu3);
    
    float F12 = FIVESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + TWOSEVENTHS*mu12*mu12;
    float F23 = FIVESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + TWOSEVENTHS*mu23*mu23;
    float F31 = FIVESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + TWOSEVENTHS*mu31*mu31;
    
    float G12 = THREESEVENTHS + 0.5*mu12*(k1/k2 + k2/k1) + FOURSEVENTHS*mu12*mu12;
    float G23 = THREESEVENTHS + 0.5*mu23*(k2/k3 + k3/k2) + FOURSEVENTHS*mu23*mu23;
    float G31 = THREESEVENTHS + 0.5*mu31*(k3/k1 + k1/k3) + FOURSEVENTHS*mu31*mu31;
    
    float Z2k12 = 0.5*d_p[1] + d_p[0]*(F12 + 0.5*mu12p*k12*(mu1/k1 + mu2/k2)) + d_p[2]*mu12p*mu12p*G12
                  + 0.5*d_p[2]*d_p[2]*mu12p*k12*mu1*mu2*(mu1/k1 + mu2/k2) 
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu12*mu12 - 1.0/3.0);
    float Z2k23 = 0.5*d_p[1] + d_p[0]*(F23 + 0.5*mu23p*k23*(mu2/k2 + mu3/k3)) + d_p[2]*mu23p*mu23p*G23
                  + 0.5*d_p[2]*d_p[2]*mu23p*k23*mu2*mu3*(mu2/k2 + mu3/k3)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu23*mu23 - 1.0/3.0);
    float Z2k31 = 0.5*d_p[1] + d_p[0]*(F31 + 0.5*mu31p*k31*(mu3/k3 + mu1/k1)) + d_p[2]*mu31p*mu31p*G31
                  + 0.5*d_p[2]*d_p[2]*mu31p*k31*mu3*mu1*(mu3/k3 + mu1/k1)
                  + 0.5*(-FOURSEVENTHS*(d_p[0] - 1.0))*(mu31*mu31 - 1.0/3.0);
                  
    float den = 1.0 + 0.5*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*(k1*k1*mu1*mu1 + k2*k2*mu2*mu2 + k3*k3*mu3*mu3)*d_p[5]*d_p[5];
    float FoG = 1.0/(den*den);
    
    return 2.0*(Z2k12*Z1k1*Z1k2*P1*P2 + Z2k23*Z1k2*Z1k3*P2*P3 + Z2k31*Z1k3*Z1k1*P3*P1)*FoG*P_L;
}

// GPU kernel to calculate the bispectrum model. This kernel uses a fixed 32-point Gaussian quadrature
// and utilizes constant and shared memory to speed things up by about 220x compared to the previous
// version of the code while improving accuracy.
__global__ void bispec_gauss_32(float3 *ks, double *Bk) {
    int tid = threadIdx.y + blockDim.x*threadIdx.x; // Block local thread ID
    
    __shared__ double int_grid[1024]; 
    
    // Calculate the value for this thread
    float phi = PI*d_xi[threadIdx.y] + PI;
    int_grid[tid] = d_wi[threadIdx.x]*d_wi[threadIdx.y]*bispec_model(threadIdx.x, phi, ks[blockIdx.x]);
    __syncthreads();
    
    // First step of reduction done by 32 threads
    if (threadIdx.y == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[tid] += int_grid[tid + i];
    }
    __syncthreads();
    
    // Final reduction and writing result to global memory done only on first thread
    if (tid == 0) {
        for (int i = 1; i < 32; ++i)
            int_grid[0] += int_grid[blockDim.x*i];
        Bk[blockIdx.x] = int_grid[0]/4.0;
    }
}

#endif
