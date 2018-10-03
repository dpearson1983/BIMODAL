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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

std::random_device seeder;
std::mt19937_64 gen(seeder());
std::uniform_real_distribution<double> dist(-1.0, 1.0);

class bkmcmc{
    int num_data, num_pars;
    std::vector<double> data; // These should have size of num_data
    std::vector<std::vector<double>> Psi; // num_data vectors of size num_data
    std::vector<double> theta_0, theta_i, param_vars, min, max; // These should all have size of num_pars
    std::vector<float4> k; // This should have size of num_data
    std::vector<bool> limit_pars; // This should have size of num_pars
    double chisq_0, chisq_i;
       
    // Sets the values of theta_i.
    void get_param_real(); // done
    
    // Calculates the chi^2 for the current proposal, theta_i
    double calc_chi_squared(); // done
    
    // Performs one MCMC trial. Returns true if proposal accepted, false otherwise
    bool trial(float4 *ks, double *Bk, double &L, double &R); // done
    
    // Writes the current accepted parameters to the screen
    void write_theta_screen(); // done
    
    // Burns the requested number of parameter realizations to move to a higher likelihood region
    void burn_in(int num_burn, float4 *ks, double *Bk); // done
    
    // Changes the initial guesses for the search range around parameters until acceptance = 0.234
    void tune_vars(float4 *ks, double *Bk); // done
    
    public:
        std::vector<double> model; // These should have size num_data
        
        // Initializes most of the data members and gets an initial chisq_0
        bkmcmc(std::string data_file, std::string cov_file, std::vector<double> &pars, 
               std::vector<double> &vars, float4 *ks, double *Bk); // done
        
        // Displays information to the screen to check that the vectors are all the correct size
        void check_init(); // done
        
        // Sets which parameters should be limited and what the limits are
        void set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in); // done
        
        // Runs the MCMC chain for num_draws realizations, writing to reals_file
        void run_chain(int num_draws, int num_burn, std::string reals_file, float4 *ks, double *Bk, bool new_chain);
        
};

void bkmcmc::get_param_real() {
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        if (bkmcmc::limit_pars[i]) {
            if (bkmcmc::theta_0[i] + bkmcmc::param_vars[i] > bkmcmc::max[i]) {
                double center = bkmcmc::max[i] - bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else if (bkmcmc::theta_0[i] - bkmcmc::param_vars[i] < bkmcmc::min[i]) {
                double center = bkmcmc::min[i] + bkmcmc::param_vars[i];
                bkmcmc::theta_i[i] = center + dist(gen)*bkmcmc::param_vars[i];
            } else {
                bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
            }
        } else {
            bkmcmc::theta_i[i] = bkmcmc::theta_0[i] + dist(gen)*bkmcmc::param_vars[i];
        }
    }
}

double bkmcmc::calc_chi_squared() {
    double chisq = 0.0;
    for (int i = 0; i < bkmcmc::num_data; ++i) {
        for (int j = i; j < bkmcmc::num_data; ++j) {
            if (bkmcmc::data[i] > 0 && bkmcmc::data[j] > 0) {
                chisq += (bkmcmc::data[i] - bkmcmc::model[i])*Psi[i][j]*(bkmcmc::data[j] - bkmcmc::model[j]);
            }
        }
    }
    return chisq;
}

bool bkmcmc::trial(float4 *ks, double *d_Bk, double &L, double &R) {
    bkmcmc::get_param_real();
    model_calc(bkmcmc::theta_i, ks, d_Bk, bkmcmc::model);
    bkmcmc::chisq_i = bkmcmc::calc_chi_squared();
    
    L = exp(0.5*(bkmcmc::chisq_0 - bkmcmc::chisq_i));
    R = (dist(gen) + 1.0)/2.0;
    
    if (L > R) {
        for (int i = 0; i < bkmcmc::num_pars; ++i)
            bkmcmc::theta_0[i] = bkmcmc::theta_i[i];
        bkmcmc::chisq_0 = bkmcmc::chisq_i;
        return true;
    } else {
        return false;
    }
}

void bkmcmc::write_theta_screen() {
    std::cout.precision(6);
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        std::cout.width(15);
        std::cout << bkmcmc::theta_0[i];
    }
    std::cout.width(15);
    std::cout << pow(bkmcmc::theta_0[3]*bkmcmc::theta_0[4]*bkmcmc::theta_0[4],1.0/3.0);
    std::cout.width(15);
    std::cout << bkmcmc::chisq_0;
    std::cout.flush();
}

void bkmcmc::burn_in(int num_burn, float4 *ks, double *d_Bk) {
    std::cout << "Burning the first " << num_burn << " trials to move to higher likelihood..." << std::endl;
    double L, R;
    for (int i = 0; i < num_burn; ++i) {
        bool move = bkmcmc::trial(ks, d_Bk, L, R);
        if (true) {
            std::cout << "\r";
            std::cout.width(5);
            std::cout << i;
            bkmcmc::write_theta_screen();
            std::cout.width(15);
            std::cout << L;
            std::cout.width(15);
            std::cout << R;
            std::cout.flush();
        }
    }
    std::cout << std::endl;
}

void bkmcmc::tune_vars(float4 *ks, double *d_Bk) {
    std::cout << "Tuning acceptance ratio..." << std::endl;
    double acceptance = 0.0;
    while (acceptance <= 0.233 || acceptance >= 0.235) {
        int accept = 0;
        double L, R;
        for (int i = 0; i < 10000; ++i) {
            bool move = bkmcmc::trial(ks, d_Bk, L, R);
            if (move) {
                std::cout << "\r";
                bkmcmc::write_theta_screen();
                accept++;
            }
        }
        std::cout << std::endl;
        acceptance = double(accept)/10000.0;
        
        if (acceptance <= 0.233) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 0.99;
        }
        if (acceptance >= 0.235) {
            for (int i = 0; i < bkmcmc::num_pars; ++i)
                bkmcmc::param_vars[i] *= 1.01;
        }
        std::cout << "acceptance = " << acceptance << std::endl;
    }
    std::ofstream fout;
    fout.open("variances.dat", std::ios::out);
    for (int i = 0; i < bkmcmc::num_pars; ++i)
        fout << bkmcmc::param_vars[i] << " ";
    fout << "\n";
    fout.close();
}

bkmcmc::bkmcmc(std::string data_file, std::string cov_file, std::vector<double> &pars, 
               std::vector<double> &vars, float4 *ks, double *d_Bk) {
    std::ifstream fin;
    std::ofstream fout;
    
    std::cout << "Reading in and storing data file..." << std::endl;
    if (std::ifstream(data_file)) {
        fin.open(data_file.c_str(), std::ios::in);
        while (!fin.eof()) {
            float4 kt;
            double B;
            fin >> kt.w >> kt.x >> kt.y >> kt.z >> B;
            if (!fin.eof()) {
                bkmcmc::k.push_back(kt);
                bkmcmc::data.push_back(B);
                bkmcmc::model.push_back(0.0);
            }
        }
        fin.close();
    } else {
        std::stringstream message;
        message << "Could not open " << data_file << std::endl;
        throw std::runtime_error(message.str());
    }
    
    bkmcmc::num_data = bkmcmc::data.size();
    std::cout << "num_data = " << bkmcmc::num_data << std::endl;
    
    gsl_matrix *cov = gsl_matrix_alloc(bkmcmc::num_data, bkmcmc::num_data);
    gsl_matrix *psi = gsl_matrix_alloc(bkmcmc::num_data, bkmcmc::num_data);
    gsl_permutation *perm = gsl_permutation_alloc(bkmcmc::num_data);
    
    std::cout << "Reading in covariance and computing its inverse..." << std::endl;
    if (std::ifstream(cov_file)) {
        fin.open(cov_file.c_str(), std::ios::in);
        for (int i = 0; i < bkmcmc::num_data; ++i) {
            for (int j = 0; j < bkmcmc::num_data; ++j) {
                double element;
                fin >> element;
                gsl_matrix_set(cov, i, j, element);
            }
        }
        fin.close();
    } else {
        std::stringstream message;
        message << "Could not open " << cov_file << std::endl;
        throw std::runtime_error(message.str());
    }
    
    int s;
    gsl_linalg_LU_decomp(cov, perm, &s);
    gsl_linalg_LU_invert(cov, perm, psi);
    
    for (int i = 0; i < bkmcmc::num_data; ++i) {
        std::vector<double> row;
        row.reserve(bkmcmc::num_data);
        for (int j = 0; j < bkmcmc::num_data; ++j) {
            row.push_back((1.0 - double(bkmcmc::num_data + 1.0)/2048.0)*gsl_matrix_get(psi, i, j));
        }
        bkmcmc::Psi.push_back(row);
    }
    
    gsl_matrix_free(cov);
    gsl_matrix_free(psi);
    gsl_permutation_free(perm);
    
    gpuErrchk(cudaMemcpy(ks, bkmcmc::k.data(), bkmcmc::num_data*sizeof(float4), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMemcpy(d_Bk, bkmcmc::model.data(), bkmcmc::num_data*sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    bkmcmc::num_pars = pars.size();
    std::cout << "num_pars = " << bkmcmc::num_pars << std::endl;
    
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        bkmcmc::theta_0.push_back(pars[i]);
        bkmcmc::theta_i.push_back(0.0);
        bkmcmc::limit_pars.push_back(false);
        bkmcmc::max.push_back(0.0);
        bkmcmc::min.push_back(0.0);
        bkmcmc::param_vars.push_back(vars[i]);
    }
    
    std::cout << "Calculating initial model and chi^2..." << std::endl;
    model_calc(bkmcmc::theta_0, ks, d_Bk, bkmcmc::model);
    bkmcmc::chisq_0 = bkmcmc::calc_chi_squared();
    
    fout.open("Bk_mod_check.dat", std::ios::out);
    for (int i =0; i < bkmcmc::num_data; ++i) {
        fout.precision(3);
        fout << bkmcmc::k[i].x << " " << bkmcmc::k[i].y << " " << bkmcmc::k[i].z << " ";
        fout.precision(15);
        fout << bkmcmc::data[i] << " " << bkmcmc::model[i] << "\n";
    }
    fout.close();
}

void bkmcmc::check_init() {
    std::cout << "Number of data points: " << bkmcmc::num_data << std::endl;
    std::cout << "    data.size()      = " << bkmcmc::data.size() << std::endl;
    std::cout << "    model.size()     = " << bkmcmc::model.size() << std::endl;
    std::cout << "    Psi.size()       = " << bkmcmc::Psi.size() << std::endl;
    std::cout << "Number of parameters:  " << bkmcmc::num_pars << std::endl;
    std::cout << "    theta_0.size()   = " << bkmcmc::theta_0.size() << std::endl;
    std::cout << "    theta_i.size()   = " << bkmcmc::theta_i.size() << std::endl;
    std::cout << "    limit_pars.size()= " << bkmcmc::limit_pars.size() << std::endl;
    std::cout << "    min.size()       = " << bkmcmc::min.size() << std::endl;
    std::cout << "    max.size()       = " << bkmcmc::max.size() << std::endl;
    std::cout << "    param_vars.size()= " << bkmcmc::param_vars.size() << std::endl;
}

void bkmcmc::set_param_limits(std::vector<bool> &lim_pars, std::vector<double> &min_in,
                              std::vector<double> &max_in) {
    for (int i = 0; i < bkmcmc::num_pars; ++i) {
        bkmcmc::limit_pars[i] = lim_pars[i];
        bkmcmc::max[i] = max_in[i];
        bkmcmc::min[i] = min_in[i];
    }
}

void bkmcmc::run_chain(int num_draws, int num_burn, std::string reals_file, float4 *ks, double *d_Bk, bool new_chain) {
    int num_old_rels = 0;
    if (new_chain) {
        std::cout << "Starting new chain..." << std::endl;
        bkmcmc::burn_in(num_burn, ks, d_Bk);
        bkmcmc::tune_vars(ks, d_Bk);
    } else {
        std::cout << "Resuming previous chain..." << std::endl;
        std::ifstream fin;
        fin.open("variances.dat", std::ios::in);
        for (int i = 0; i < bkmcmc::num_pars; ++i) {
            double var;
            fin >> var;
            bkmcmc::param_vars[i] = var;
        }
        fin.close();
        fin.open(reals_file.c_str(), std::ios::in);
        while (!fin.eof()) {
            double alpha;
            num_old_rels++;
            std::cout << "\r";
            for (int i = 0; i < bkmcmc::num_pars; ++i) {
                fin >> bkmcmc::theta_0[i];
                std::cout.width(10);
                std::cout << bkmcmc::theta_0[i];
            }
            fin >> alpha;
            fin >> bkmcmc::chisq_0;
            std::cout.width(10);
            std::cout << alpha;
            std::cout.width(10);
            std::cout << bkmcmc::chisq_0;
        }
        fin.close();
        num_old_rels--;
    }
    
    std::ofstream fout;
    double L, R;
    fout.open(reals_file.c_str(), std::ios::app);
    fout.precision(15);
    for (int i = 0; i < num_draws; ++i) {
        bool move = bkmcmc::trial(ks, d_Bk, L, R);
        for (int par = 0; par < bkmcmc::num_pars; ++par) {
            fout << bkmcmc::theta_0[par] << " ";
        }
        double alpha = pow(bkmcmc::theta_0[3]*bkmcmc::theta_0[4]*bkmcmc::theta_0[4], 1.0/3.0);
        fout << alpha << " " << bkmcmc::chisq_0 << "\n";
        if (move) {
            std::cout << "\r";
            std::cout.width(15);
            std::cout << i + num_old_rels;
            bkmcmc::write_theta_screen();
        }
    }
    std::cout << std::endl;
    fout.close();
}
    
#endif
