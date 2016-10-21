// #include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "minibatch_baseline.h"
#include "utils/mnist_utils.hpp"

/* This is a baseline for stochastic gradient descent using mini-batches. In this version,
the mini-batch parameters are updated only once in an epoch until they all converge. This version runs faster
than when each mini-batch goes through all the epochs before the next mini-batch is computed. */


template <typename Type1, typename Type2, typename Type3, typename Type4>
static void nBlas_matrixVectorMultiply(Type1* matrix, 
                                       Type2* vect, 
                                       Type3 scalar, 
                                       size_t num_data_points, 
                                       size_t num_features, 
                                       Type4* result) {
    // non-blas function for matrixVectorMultiplication
    memset(result, 0, num_features * sizeof(Type4));
    for (size_t i = 0; i < num_data_points; i++) {
        addVectors(result, &matrix[i * num_features], num_features, scalar * vect[i]);
    }
}

//overload operator to cantenate two vector (more efficient one)
template <typename T>
std::vector<T> &operator+=(std::vector<T> &A, const std::vector<T> &B)
{
    A.reserve( A.size() + B.size() );                // preallocate memory without erase original data
    A.insert( A.end(), B.begin(), B.end() );         // add B;
    return A;                                        // here A could be named AB
}


static void gradientForMiniBatch(
    DataSet training_set,
    FeatureType* gradient,
    size_t batch_size,
    size_t starting_point) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType));

    float* probabilities_of_positive = new float[batch_size];
    size_t idx = 0;

    //computes logistic function for each data point in the mini-batch
    for (int i= starting_point; i < starting_point + batch_size; i++) {
        idx = i * training_set.num_features;
        probabilities_of_positive[i % batch_size] 
            = logisticFunction(training_set.parameter_vector, 
                               &training_set.data_points[idx], 
                               training_set.num_features);
    }

    // computes difference between predicted probability and actual label (PI - Y)
    addVectors(probabilities_of_positive, 
               &training_set.labels[starting_point], 
               batch_size, -1);

    // finishes computation of gradient: (1/n) * X^T * (PI(theta, X) - YI)
    float factor = 1.0f/batch_size;
    size_t starting_point_idx = starting_point *  training_set.num_features;
    matrixVectorMultiply(&training_set.data_points[starting_point_idx], 
                         probabilities_of_positive, factor, batch_size, 
                         training_set.num_features, gradient);

    delete[] probabilities_of_positive;
}

static void softmaxGradientForMiniBatch(
    DataSet training_set,
    FeatureType* gradient,
    size_t batch_size,
    size_t starting_point) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType) * LABEL_CLASS);


    std::vector<float> probabilities_of_each;
    size_t idx = 0;
    for (size_t i = starting_point; i < starting_point + batch_size; i++) {

        idx = i * training_set.num_features;
        //http://stackoverflow.com/questions/3177241/what-is-the-best-way-to-concatenate-two-vectors
        //catenate vectors
        probabilities_of_each+=softmaxFunctionFloat(
                                                training_set.parameter_vector,
                                                &training_set.data_points[idx],
                                                training_set.num_features);
    }
    //convert vector to an array
    float* probabilities_array = &probabilities_of_each[0];

    //establish groundTruth array UFLDL Tutorial
    std::vector<float> groundTruth(LABEL_CLASS * batch_size);
    for (size_t i = starting_point; i < starting_point + batch_size; i++) {

        int idx1 = training_set.labels[i];
        groundTruth[idx1 + (i-starting_point) * LABEL_CLASS]=1.0f;
    }

    float* groundTruth_array = &groundTruth[0];
    addVectors(probabilities_array,
               groundTruth_array,
               batch_size * LABEL_CLASS,
               -1);
    // finishes computation of gradient: (1/n) * X^T * (PI(theta, X) - YI)
    float factor = 1.0f/batch_size;
    size_t starting_point_idx = starting_point *  training_set.num_features;

    matrixMatrixMultiply(probabilities_array,
                         &training_set.data_points[starting_point_idx],
                         factor,
                         batch_size,
                         training_set.num_features,
                         gradient);
}

static void softmaxBoldDriver(
    DataSet training_set,
    FeatureType* gradient,
    double* step_size) {

    double previous_loss = softmaxLossFunction(training_set);

    updateParameters(training_set.parameter_vector,
                     gradient,
                     training_set.num_features,
                     *step_size);

    double current_loss = softmaxLossFunction(training_set);

    // if it's going in the right direction, increase step size
    if (current_loss < previous_loss) {
        *step_size *= 1.045;
    }
    // if the previous step was too big and the loss increased,
    // revert step and reduce step size
    else {
        bool revert = true;
        int num_reverts = 0, max_reverts = 10;
        while (revert && (num_reverts < max_reverts)) {
            updateParameters(training_set.parameter_vector,
                             gradient,
                             training_set.num_features,
                             *step_size,
                             revert);

            *step_size *= 0.5;

            updateParameters(training_set.parameter_vector,
                             gradient,
                             training_set.num_features,
                             *step_size);

             current_loss = softmaxLossFunction(training_set);

             revert = (current_loss > previous_loss);
        }

    }
}

void trainMiniBatchGradientDescent(
    DataSet training_set,
    TrainingOptions training_options) {

    FeatureType* gradient = new FeatureType[training_set.num_features * LABEL_CLASS];

    // read configuration parameters
    double step_size = *training_options.step_size;



    const double characteristic_time =
                (fieldExists(training_options.config_params, "characteristic_time"))
                ? training_options.config_params["characteristic_time"]
                : CHARACTERISTIC_TIME;

    size_t curr_num_epochs =
                (fieldExists(training_options.config_params, "curr_num_epochs"))
                ? training_options.config_params["curr_num_epochs"]
                : 0;

    const size_t batch_size =
            (fieldExists(training_options.config_params, "batch_size"))
            ? training_options.config_params["batch_size"]
            : BATCH_SIZE; 
    
    size_t num_mini_batches = training_set.num_data_points / batch_size;

    
    
    double annealed_step_size = step_size;

    for (int i = 0; i < training_options.num_epochs; i++){
         annealed_step_size = training_options.config_params["initial_step_size"]
                             / (1.0
                                + (curr_num_epochs
                                   * training_set.num_data_points
                                   / characteristic_time));
        curr_num_epochs++;
        
        for (int j = 0; j < num_mini_batches; j++){
            size_t idx = j * batch_size;
            // compute gradient
            softmaxGradientForMiniBatch(training_set, gradient, batch_size, idx);

            // softmaxBoldDriver(training_set, gradient, &step_size);
            

            // update parameters
            updateParameters(training_set.parameter_vector, gradient, 
                training_set.num_features, annealed_step_size);

        }             
        // double previous_loss = softmaxLossFunction(training_set);
        
         *training_options.step_size = annealed_step_size;
        
    }

    delete[] gradient;
}
