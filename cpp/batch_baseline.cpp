#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "batch_baseline.h"
#include "utils/mnist_utils.hpp"

//overload operator to cantenate two vector
template <typename T>
std::vector<T> operator+(const std::vector<T> &A, const std::vector<T> &B)
{
    std::vector<T> AB;
    AB.reserve( A.size() + B.size() );                // preallocate memory
    AB.insert( AB.end(), A.begin(), A.end() );        // add A;
    AB.insert( AB.end(), B.begin(), B.end() );        // add B;
    return AB;
}
//overload operator to cantenate two vector (more efficient one)
template <typename T>
std::vector<T> &operator+=(std::vector<T> &A, const std::vector<T> &B)
{
    A.reserve( A.size() + B.size() );                // preallocate memory without erase original data
    A.insert( A.end(), B.begin(), B.end() );         // add B;
    return A;                                        // here A could be named AB
}



static void softmaxGradientForWholeBatch(
    DataSet training_set,
    FeatureType* gradient) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType) * LABEL_CLASS);

    // computes softmax function for each data point in the training set
    std::vector<float> probabilities_of_each;
    size_t idx = 0;
    for (size_t i = 0; i < training_set.num_data_points; i++) {

        idx = i * training_set.num_features;
        //http://stackoverflow.com/questions/3177241/what-is-the-best-way-to-concatenate-two-vectors
        // concatenate-two-vectors to make probability vector 
        probabilities_of_each+=softmaxFunctionFloat(
                                                training_set.parameter_vector,
                                                &training_set.data_points[idx],
                                                training_set.num_features);
    }
    //change a vector to an array
    //http://stackoverflow.com/questions/2923272/how-to-convert-vector-to-array-c
    float* probabilities_array = &probabilities_of_each[0];
    //initialize to 0 by c++ 
    std::vector<float> groundTruth(LABEL_CLASS * training_set.num_data_points);
    //establish groundtruth function to see softmaxExercise at UFLDL Tutorial 
    for (size_t i = 0; i < training_set.num_data_points; i++) {
        int idx1 = training_set.labels[i];
        groundTruth[idx1 + i * LABEL_CLASS]=1.0f;
    }



    //change a vector to an array
    float* groundTruth_array = &groundTruth[0];
    //computing 1{P(y=k|x,theta)-1{y=k}}
    addVectors(probabilities_array,
               groundTruth_array,
               training_set.num_data_points * LABEL_CLASS,
               -1);


    float factor = 1.0f / training_set.num_data_points;
    //finish computing x*sigma (i=1..k){P(y=k|x,theta)-1{y=k}}
    matrixMatrixMultiply(probabilities_array,
                         training_set.data_points,
                         factor,
                         training_set.num_data_points,
                         training_set.num_features,
                         gradient);

}
// computes gradient for the whole training set
static void gradientForWholeBatch(
    DataSet training_set,
    FeatureType* gradient) {

    memset(gradient, 0, training_set.num_features * sizeof(FeatureType));

    float* probabilities_of_positive = new float[training_set.num_data_points];

    // computes logistc function for each data point in the training set
    size_t idx = 0;
    for (size_t i = 0; i < training_set.num_data_points; i++) {

        idx = i * training_set.num_features;

        probabilities_of_positive[i] =  logisticFunction(
                                                training_set.parameter_vector,
                                                &training_set.data_points[idx],
                                                training_set.num_features);
    }

    // computes difference between
    // predicted probability and actual label: (PI - Y)
    addVectors(probabilities_of_positive,
               training_set.labels,
               training_set.num_data_points,
               -1);

    // finishes computation of gradient: (1/n) * X^T * (PI(theta, X) - YI)
    float factor = 1.0f / training_set.num_data_points;
    matrixVectorMultiply(training_set.data_points,
                         probabilities_of_positive,
                         factor,
                         training_set.num_data_points,
                         training_set.num_features,
                         gradient);

    delete[] probabilities_of_positive;
}



// Bold Driver: adjusting the step size according to the result of the
// last step and reverting the step if results are worse than they were before.
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

// Bold Driver: adjusting the step size according to the result of the
// last step and reverting the step if results are worse than they were before.
static void boldDriver(
    DataSet training_set,
    FeatureType* gradient,
    double* step_size) {

    double previous_loss = lossFunction(training_set);

    updateParameters(training_set.parameter_vector,
                     gradient,
                     training_set.num_features,
                     *step_size);

    double current_loss = lossFunction(training_set);

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

             current_loss = lossFunction(training_set);

             revert = (current_loss > previous_loss);
        }

    }
}


// uses simulated annealing to update step size
static void anneal(
    size_t current_epoch,
    double initial_step_size,
    double* annealed_step_size,
    DataSet training_set,
    FeatureType* gradient) {

    double characteristic_time = CHARACTERISTIC_TIME;

    (*annealed_step_size) = initial_step_size
                          / (1.0 + (current_epoch / characteristic_time));

    updateParameters(training_set.parameter_vector,
                   gradient,
                   training_set.num_features,
                   *annealed_step_size
                   );
}


// executes serial implementation of stochastic gradient descent for
// softmax regression until convergence or for a fixed number of epochs
void trainBatchGradientDescent(
    DataSet training_set,
    TrainingOptions training_options) {

    // read configuration parameters
    double step_size = *training_options.step_size;
    double initial_step_size = step_size;

    const double tolerance =
            (fieldExists(training_options.config_params, "tolerance"))
            ? training_options.config_params["tolerance"]
            : TOLERANCE;

    FeatureType* gradient = new FeatureType[training_set.num_features * LABEL_CLASS];

    // initial loss
    
    
    // double current_loss;
    // double initial_step_size = step_size;

    for (size_t epoch = 0; epoch < training_options.num_epochs; epoch++) {
        // compute gradient and update parameters

        softmaxGradientForWholeBatch(training_set, gradient);
        

        softmaxBoldDriver(training_set, gradient, &step_size);
        
        // anneal(epoch, initial_step_size, &step_size, training_set, gradient);


        // updateParameters(training_set.parameter_vector,
        //            gradient,
        //            training_set.num_features,
        //            initial_step_size);

        // stop iterating if the gradient is close to zero
        if (norm2(gradient, training_set.num_features * LABEL_CLASS) < tolerance)
            break;
    }



    *training_options.step_size = step_size;

    delete[] gradient;
}
