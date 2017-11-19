#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cublas_v2.h>

#include "sgd_cublas.h"
#include "mnist_utils_cuda.cuh"


static FeatureType *d_parameter_vector, *d_data_points, *d_gradient, *d_result;
static cublasHandle_t handle;

static void setCudaVariables(
    size_t num_features,
    size_t num_data_points,
    FeatureType* data_points){
    // FeatureType* parameter_vector) {

    checkCuBlasErrors(cublasCreate(&handle));

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    const size_t size_of_datapoint = num_features * sizeof(FeatureType);

    checkCudaErrors(cudaMalloc(&d_parameter_vector, LABEL_CLASS * size_of_datapoint));
    checkCudaErrors(cudaMalloc(&d_gradient, LABEL_CLASS * size_of_datapoint));
    checkCudaErrors(cudaMalloc(&d_data_points, (size_of_datapoint
                                                * num_data_points)));
    // cublas matrix multiplication require
    checkCudaErrors(cudaMalloc(&d_result, LABEL_CLASS * sizeof(FeatureType)));

    // printf("size_t %d..\n",LABEL_CLASS * size_of_datapoint);

    checkCudaErrors(cudaMemcpy(d_data_points,
                               data_points,
                               (size_of_datapoint * num_data_points),
                               cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy(d_parameter_vector, 
    //                            parameter_vector,
    //                            LABEL_CLASS * num_features * sizeof(FeatureType),
    //                             cudaMemcpyHostToDevice));

    // printf("after copy parameter..\n");
//
 }

static void cleanUp() {

    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_gradient));
    checkCudaErrors(cudaFree(d_data_points));
    //cublas matrix multiplication require
    checkCudaErrors(cudaFree(d_result));

    cublasDestroy(handle);
}

// computes gradient for a single datapoint using cublas
static void p_gradientForSinglePoint (
    cublasHandle_t handle,
    FeatureType* d_parameter_vector,
    FeatureType* d_data_point_i,
    FeatureType* d_result,
    LabelType label,
    size_t num_features,
    FeatureType* d_gradient){

    // float probability_of_positive = p_logisticFunction(
    //     handle,
    //     d_parameter_vector,
    //     d_data_point_i,
    //     num_features);

    float probabilities_of_each[LABEL_CLASS] = {0};

    // for(size_t i=0 ; i< LABEL_CLASS; i++){
    //     printf("p is %f\n",probabilities_of_each[i]);
    // } 

    // exit(1);

    // p_softmaxFunction2(handle,
    //     d_parameter_vector,
    //     d_data_point_i,
    //     probabilities_of_each,
    //     num_features,
    //     LABEL_CLASS);


    p_softmaxFunction(handle,
        d_parameter_vector,
        d_data_point_i,
        d_result,
        probabilities_of_each,
        num_features,
        LABEL_CLASS);

    // for(size_t i=0; i<LABEL_CLASS; i++){
    //     printf("posibiility_each is %f\n",probabilities_of_each[i]);
    // }

    
    // checkCudaErrors(cudaGetLastError());

    checkCudaErrors(
        cudaMemset(d_gradient, 0, LABEL_CLASS * num_features * sizeof(FeatureType)));

    // for(size_t i= 0; i< num_features; i++){
    //     printf("g is %f\n",d_gradient[i]);
    // }
    // exit(1);
    for(size_t i=0; i<LABEL_CLASS; i++){
        //case one parameter with the same label
        if(label==i){
            // addVectors((&gradient[i*num_features]), 
            //            data_point,
            //            num_features,
            //            (posibiility_each[i] - 1));
            p_add_vectors(
                    handle,
                    &d_gradient[i*num_features],
                    d_data_point_i,
                    num_features,
                    (probabilities_of_each[i] - 1));
        }
        //case two not the same label
        else{
            // addVectors((&gradient[i*num_features]), 
            //            data_point,
            //            num_features,
            //            (posibiility_each[i]));
            p_add_vectors(
                    handle,
                    &d_gradient[i*num_features],
                    d_data_point_i,
                    num_features,
                    probabilities_of_each[i]);
        }

    }


    // for(size_t i= 5*num_features; i< 6*num_features; i++){
    //     printf("posibiility_each is %f\n",d_gradient[i]);
    // }
    // exit(1);
    // p_add_vectors(
    //     handle,
    //     d_gradient,
    //     d_data_point_i,
    //     num_features,
    //     (probability_of_positive - label));
    // delete[] probabilities_of_each;
}

// executes serial implementation of stochastic gradient descent
// for logistic regression with a fixed number of iterations
// config_params: {step_size}
void trainStochasticGradientDescent3(
    DataSet training_set,
    TrainingOptions training_options){

    setCudaVariables(
        training_set.num_features,
        training_set.num_data_points,
        training_set.data_points);

    //FeatureType* gradient = new FeatureType[training_set.num_features];

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

    double annealed_step_size = step_size;

    printf("%d\n",training_options.num_epochs);



    for (size_t k = 0; k < training_options.num_epochs; k++) {

        printf("run number %d\n",k);

        // simulated annealing (reduces step size as it converges)
        annealed_step_size = training_options.config_params["initial_step_size"]
                             / (1.0
                                + (curr_num_epochs
                                   * training_set.num_data_points
                                   / characteristic_time));
        curr_num_epochs++;

        for (size_t i = 0; i < training_set.num_data_points; i++) {
            // printf("i is %d\n",i);
            FeatureType* d_data_point_i = &d_data_points[i * training_set.num_features];
            // checkCudaErrors(cudaMemcpy(d_parameter_vector, training_set.parameter_vector,LABEL_CLASS * training_set.num_features * sizeof(FeatureType), cudaMemcpyHostToDevice));
            p_gradientForSinglePoint(handle, d_parameter_vector, d_data_point_i, d_result, training_set.labels[i], training_set.num_features, d_gradient);
            p_updateParameters(handle, d_parameter_vector, d_gradient, LABEL_CLASS * training_set.num_features, annealed_step_size);
        }
    }
    checkCudaErrors(cudaMemcpy(training_set.parameter_vector, d_parameter_vector, (LABEL_CLASS * training_set.num_features * sizeof(FeatureType)), cudaMemcpyDeviceToHost));
    //delete[] gradient;

    *training_options.step_size = annealed_step_size;

    cleanUp();
}
