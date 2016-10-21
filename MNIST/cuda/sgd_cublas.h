#ifndef MNIST_SGD_CUBLAS
#define MNIST_SGD_CUBLAS

/*
 * Parallel approach #1:
 * Using OpenBLAS for regular stochastic gradient descent
 */

#include "utils/mnist_defs.h"

void trainStochasticGradientDescent3(DataSet training_set, TrainingOptions training_options);

#endif
