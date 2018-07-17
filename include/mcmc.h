/* mcmc.h
 * David W. Pearson
 * 17 July 2018
 * 
 * This header file will be responsible for running Markov Chain Monte Carlo fitting.
 */

#ifndef _MCMC_H_
#define _MCMC_H_

#include "bispectrum_model.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>

class bkmcmc{
    int num_data, num_pars;
    std::vector<double> data, Bk_mono, Bk_quad; // These should have size of num_data
    std::vector<std::vector<double>> Psi; // num_data vectors of size num_data
    std::vector<double> theta_0, theta_i, param_vars, min, max; // These should all have size of num_pars
    std::vector<float3> k; // This should have size of num_data
    std::vector<bool> limit_pars; // This should have size of num_pars
    double chisq_0, chisq_i;
    
    // Calculates the model bispectra for the input parameters, pars.
    void model_calc(std::vector<double> &pars, float3 *ks, double *Bk); // done
    
    // Sets the values of theta_i.
    void get_param_real(); // done
    
    // Calculates the chi^2 for the current proposal, theta_i
    double calc_chi_squared(); // done
    
    // Performs one MCMC trial. Returns true if proposal accepted, false otherwise
    bool trial(float3 *ks, double *Bk, double &L, double &R); // done
    
    // Writes the current accepted parameters to the screen
    void write_theta_screen(); // done
    
    // Burns the requested number of parameter realizations to move to a higher likelihood region
    void burn_in(int num_burn, float3 *ks, double *Bk); // done
    
    // Changes the initial guesses for the search range around parameters until acceptance = 0.234
    void tune_vars(float3 *ks, double *Bk); // done
    
    public:
        // Initializes most of the data members and gets an initial chisq_0
        bkmcmc(std::string data_file, std::string cov_file, std::vector<double> &pars, 
               std::vector<double> &vars, float3 *ks, double *Bk); // done
        
        // Displays information to the screen to check that the vectors are all the correct size
        void check_init(); // done
        
        void 
        
        // Sets which parameters should be limited and what the limits are
        void set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in); // done
        
        // Runs the MCMC chain for num_draws realizations, writing to reals_file
        void run_chain(int num_draws, int num_burn, std::string reals_file, float3 *ks, double *Bk, bool new_chain);
        
};
