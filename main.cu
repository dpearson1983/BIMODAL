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
#include "include/harppi.h"
#include "include/make_spline.h"
#include "include/pk_slope.h"

int main(int argc, char *argv[]) {
    // Use HARPPI hidden in an object file to parse parameters
    parameters p(argv[1]);
    
    // Generate cubic splines of the input BAO and NW power spectra
    std::vector<float4> Pk = make_spline(p.gets("input_power"));
    
    // Copy the splines to the allocated GPU memory
    gpuErrchk(cudaMemcpyToSymbol(d_Pk, Pk.data(), 128*sizeof(float4)));
    
    // Copy Gaussian Quadrature weights and evaluation point to GPU constant memory
    gpuErrchk(cudaMemcpyToSymbol(d_wi, &w_i[0], 32*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_xi, &x_i[0], 32*sizeof(float)));
    
    // Declare a pointer for the integration workspace and allocate memory on the GPU
    double *d_Bk;
    float4 *d_ks;
    
    gpuErrchk(cudaMalloc((void **)&d_Bk, p.geti("num_data")*sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_ks, p.geti("num_data")*sizeof(float4)));
    
    std::vector<double> start_params;
    std::vector<bool> limit_params;
    std::vector<double> var_i;
    std::vector<double> min;
    std::vector<double> max;
    for (int i = 0; i < p.geti("num_params"); ++i) {
        start_params.push_back(p.getd("start_params", i));
        limit_params.push_back(p.getb("limit_params", i));
        var_i.push_back(p.getd("vars", i));
        min.push_back(p.getd("min_params", i));
        max.push_back(p.getd("max_params", i));
    }
    
    // Initialize bkmcmc object
    bkmcmc bk_fit(p.gets("data_file"), p.gets("cov_file"), start_params, var_i, d_ks, d_Bk);
    
    // Check that the initialization worked
    bk_fit.check_init();
    
    // Set any limits on the parameters
    bk_fit.set_param_limits(limit_params, min, max);
    
    // Run the MCMC chain
    bk_fit.run_chain(p.geti("num_draws"), p.geti("num_burn"), p.gets("reals_file"), d_ks, d_Bk,
                     p.getb("new_chain"));
    
    // Free device pointers
    gpuErrchk(cudaFree(d_Bk));
    gpuErrchk(cudaFree(d_ks));
    
    return 0;
}
