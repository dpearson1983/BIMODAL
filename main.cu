/* BIMODAL v1
 * David W. Pearson
 * September 28, 2017
 * 
 * This version of the code will implement some improvements to make the model better fit non-linear
 * features present in the data. The algorithm is effectively that of Gil-Marin 2012/2015.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include "include/gpuerrchk.h"
#include "include/mcmc.h"
#include "include/hide_harppi.h"
#include "include/make_spline.h"
#include "include/pk_slope.h"

int main(int argc, char *argv[]) {
    // Use HARPPI hidden in an object file to parse parameters
    mcmc_parameters p(argv[1]);
    
    // Generate cubic splines of the input BAO and NW power spectra
    std::vector<float4> Pk = make_spline(p.input_power);
    
    // Copy the splines to the allocated GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_Pk, Pk.data(), 128*sizeof(float4)));
    
    // Copy Gaussian Quadrature weights and evaluation point to GPU constant memory
    gpuErrchk(cudaMemcpyToSymbol(d_wi, &w_i[0], 32*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_xi, &x_i[0], 32*sizeof(float)));
    
    // Declare a pointer for the integration workspace and allocate memory on the GPU
    double *d_Bk;
    float3 *d_ks;
    
    gpuErrchk(cudaMalloc((void **)&d_Bk, p.num_data*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_ks, p.num_data*sizeof(float3)));
    
    // Initialize bkmcmc object
    bkmcmc bk_fit(p.data_file, p.cov_file, p.start_params, p.var_i, d_ks, d_Bk);
    
    // Check that the initialization worked
    bk_fit.check_init();
    
    // Set any limits on the parameters
    bk_fit.set_param_limits(p.limit_params, p.min, p.max);
    
    // Run the MCMC chain
    bk_fit.run_chain(p.num_draws, p.num_burn, p.reals_file, d_ks, d_Bk, p.new_chain);
    
    // Free device pointers
    gpuErrchk(cudaFree(d_Bk));
    gpuErrchk(cudaFree(d_ks));
    
    return 0;
}
