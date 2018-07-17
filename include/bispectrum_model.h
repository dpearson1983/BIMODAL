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

