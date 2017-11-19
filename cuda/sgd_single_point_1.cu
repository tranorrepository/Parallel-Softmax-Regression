#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "sgd_single_point_1.h"
#include "mnist_utils_cuda.cuh"


/*
 * Parallel approach to Stochastic Gradient Descent #2 - CUDA:
 * Parallel computation of gradient and synchronous update of the parameter
 * vector with atomic operations. Several threads work on every data point, and
 * several datapoints are processed concurrently.
 *
 * Note: This version only works when dimensions of variables are powers of 2
 */


// pointers to device global variables
static FeatureType *d_parameter_vector, *d_data_points;
static LabelType *d_labels;


// Allocate space for the data set, labels and parameter vector in global memory
// Then, copy values for those host variables to the device variables
static void setCudaVariables(
    size_t num_features,
    size_t num_data_points,
    FeatureType* data_points,
    LabelType* labels,
    FeatureType* parameter_vector) {

    checkCudaErrors(cudaMalloc(
        &d_parameter_vector, LABEL_CLASS * num_features * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(
        &d_data_points, num_data_points * num_features * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(
        &d_labels, num_data_points * sizeof(LabelType)));

    checkCudaErrors(cudaMemcpy(
        d_data_points,
        data_points,
        num_data_points * num_features * sizeof(FeatureType),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_labels,
        labels,
        num_data_points * sizeof(LabelType),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_parameter_vector,
        parameter_vector,
        LABEL_CLASS * num_features * sizeof(FeatureType),
        cudaMemcpyHostToDevice));
}

// Free space in global memory
static void cleanUp() {
    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_data_points));
    checkCudaErrors(cudaFree(d_labels));
}


// calcuate partitial matrix-vector product
// thread is divide in several classes
// each number in one class calcualte strided
// dot product (later one refers to spam-filter) 
static __device__ void d_partialMatrixVectorProduct(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    FeatureType* shared_memory,
    size_t num_features,
    size_t threads_per_datapoint,
    size_t threads_class_per_datapoint) {
    //memset to 0
    FeatureType partial_dot = 0;

    size_t thread_offset = threadIdx.x % threads_per_datapoint;

    size_t num_thread_each_class = threads_per_datapoint / threads_class_per_datapoint;
    size_t relative_tidx_each_class = thread_offset % num_thread_each_class;
    size_t parameters_idx_each_class =  thread_offset / num_thread_each_class;

    // calculate parameter in parallel by several class of threads
    for (size_t j = relative_tidx_each_class; j < num_features; j += num_thread_each_class)
        partial_dot += data_point_i[j] * parameter_vector[j + parameters_idx_each_class * num_features];

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x] = partial_dot;
}

// updates parameter vector in parallel when N threads are working on each point
// Each time update one parameter
// static __device__ void d_updateParameters(
//     FeatureType* data_point_i,
//     FeatureType* parameter_vector,
//     size_t num_features,
//     size_t threads_per_datapoint,
//     size_t point_idx_in_block,
//     size_t relative_tidx,
//     FeatureType* step_size_times_prob_i_minus_label_i) {

//     // printf("enter update parameters in sgd_single_point\n");

//     size_t thread_offset = threadIdx.x % threads_per_datapoint;
//     // __syncthreads();

//     for(size_t i= 0;i < LABEL_CLASS; i++){
        
//         for (size_t j = thread_offset; j < num_features; j += threads_per_datapoint){

//             atomicAdd(&parameter_vector[j+i*num_features], - data_point_i[j] 
//                 * step_size_times_prob_i_minus_label_i[point_idx_in_block * LABEL_CLASS+i]);

//         }
        
//     }        
// }   
// update parameter for all kinds of blocking
// slightly faster than above one
// thread is divide in several classes
// each number in one class calcualte strided
// vector add 
static __device__ void d_updateParameters(
    FeatureType* data_point_i,
    FeatureType* parameter_vector,
    size_t num_features,
    size_t threads_per_datapoint,
    size_t point_idx_in_block,
    size_t relative_tidx,
    size_t threads_class_per_datapoint,
    FeatureType* step_size_times_prob_i_minus_label_i) {

    // printf("enter update parameters in sgd_single_point\n");

    size_t thread_offset = threadIdx.x % threads_per_datapoint;
    size_t num_thread_each_class = threads_per_datapoint / threads_class_per_datapoint;
    size_t relative_tidx_each_class = thread_offset % num_thread_each_class;
    size_t parameters_idx_each_class =  thread_offset / num_thread_each_class;
    size_t num_parameter_each_class = LABEL_CLASS / threads_class_per_datapoint;
    // __syncthreads();

    for(size_t i = 0; i < num_parameter_each_class ; i++){
        
        for (size_t j = relative_tidx_each_class; j < num_features; j += num_thread_each_class){

            size_t parameters_idx = parameters_idx_each_class +  threads_class_per_datapoint * i;
//            size_t probability_idx = threads_class_per_datapoint * i + parameters_idx_each_class;

            atomicAdd(&parameter_vector[j+parameters_idx * num_features], - data_point_i[j] 
                * step_size_times_prob_i_minus_label_i[point_idx_in_block * LABEL_CLASS + parameters_idx]);

        }
        
    }        
} 

// Kernel for Parallel Stochastic Gradient Descent in CUDA using
// shared parameter vector
static __global__ void p_SgdWithSharedParameterVector(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    LabelType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t threads_per_datapoint,
    size_t threads_class_per_datapoint,
    double step_size) {

    extern __shared__ FeatureType shared_memory[];

    
    // memory place for possibility
    float *probabilities_of_each = (float*)&shared_memory[blockDim.x]; 
    // computes several indexes, offsets and strides to simplify further code
    size_t tidx = threadIdx.x;
    size_t num_parameter_each_class = LABEL_CLASS / threads_class_per_datapoint;
    size_t points_per_block = (blockDim.x / threads_per_datapoint);
    size_t point_idx = (blockIdx.x * points_per_block)
                     + (tidx / threads_per_datapoint);
    // index relative to the datapoint instead of the block
    size_t relative_tidx = tidx % threads_per_datapoint;
    size_t point_idx_in_shmem = tidx - relative_tidx;
    size_t point_idx_in_block = tidx / threads_per_datapoint;
    // index relative to each class of thread
    size_t num_thread_each_class = threads_per_datapoint / threads_class_per_datapoint;
    size_t relative_tidx_each_class = relative_tidx % num_thread_each_class;
    // size_t parameters_idx_each_class =  relative_tidx / num_thread_each_class;


    FeatureType* data_point_i = NULL;

    // make sure the threads don't go out of bounds
    if (point_idx < num_data_points) {

        data_point_i = (FeatureType*) &data_points[point_idx * num_features];

        // compute partial matrix-vector product
        for(size_t i = 0; i < num_parameter_each_class; i++){
            
           
            d_partialMatrixVectorProduct(
                data_point_i,
                &parameter_vector[i * threads_class_per_datapoint * num_features],
                shared_memory,
                num_features,
                threads_per_datapoint,
                threads_class_per_datapoint);

            __syncthreads();

            // sum-reduce the results of partial dot product to get final result        
            for (size_t s = num_thread_each_class / 2; s > 0; s>>=1) {
                
                if (relative_tidx_each_class < s) {
                    shared_memory[tidx] += shared_memory[tidx+s];
                }
                __syncthreads();
            }

            //copy result from shared_memory to possibility_each
            //at the same time take fast exponential
            if(relative_tidx < threads_class_per_datapoint){
                // idx to find where the sum of dot product lies
                // size_t block_idx = relative_tidx / threads_class_per_datapoint;
                size_t sub_block_idx = relative_tidx % threads_class_per_datapoint;

                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx 
                    + i * threads_class_per_datapoint]= 
                        __expf(shared_memory[sub_block_idx * num_thread_each_class + point_idx_in_shmem]);

            }
            __syncthreads();
        
        }

        d_softMaxFunction(probabilities_of_each,
                    relative_tidx,
                    point_idx_in_block);
        
        //calculate step_size_times_prob_i_minus_label_i, store in the same position
        //calculate eta * {y(i)=k}−P(y(i)=k|x(i)
        if(relative_tidx < LABEL_CLASS){
            if(labels[point_idx]==relative_tidx){
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] -= 1;
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= step_size;
            }else{                   
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= step_size;
            }
        }
        
        __syncthreads();
       
        // update parameter
        d_updateParameters(
            data_point_i,
            parameter_vector,
            num_features,
            threads_per_datapoint,
            point_idx_in_block,
            relative_tidx,
            threads_class_per_datapoint,
            probabilities_of_each);

     
    }   

}


// Executes serial implementation of stochastic gradient descent for softmax
// regression with a fixed number of iterations
// config_params: {step_size, threads_per_datapoint, datapoints_per_block, 
// threads_class_per_datapoint}
void trainParallelStochasticGradientDescent1(
    DataSet training_set,
    TrainingOptions training_options) {

    // shuffle datapoints in order to add more stochasticity
    // shuffleKeyValue(
    //     training_set.data_points,
    //     training_set.labels,
    //     training_set.num_data_points,
    //     training_set.num_features);

    // allocate device memory
    setCudaVariables(
        training_set.num_features,
        training_set.num_data_points,
        training_set.data_points,
        training_set.labels,
        training_set.parameter_vector);


    // read configuration parameters and initialize grid and block dimensions
    double step_size = *training_options.step_size;

    const double threads_per_datapoint =
            (fieldExists(training_options.config_params, "threads_per_datapoint"))
            ? training_options.config_params["threads_per_datapoint"]
            : THREADS_PER_DATAPOINT;

    const double datapoints_per_block =
            (fieldExists(training_options.config_params, "datapoints_per_block"))
            ? training_options.config_params["datapoints_per_block"]
            : DATAPOINTS_PER_BLOCK;

    const double characteristic_time =
            (fieldExists(training_options.config_params, "characteristic_time"))
            ? training_options.config_params["characteristic_time"]
            : CHARACTERISTIC_TIME;

    const double threads_class_per_datapoint =
            (fieldExists(training_options.config_params, "threads_class_per_datapoint"))
            ? training_options.config_params["threads_class_per_datapoint"]
            : THREADS_CLASS_PER_DATAPOINT;

    size_t curr_num_epochs =
            (fieldExists(training_options.config_params, "curr_num_epochs"))
            ? training_options.config_params["curr_num_epochs"]
            : 0;

    double annealed_step_size = step_size;

    const dim3 block_size(
        threads_per_datapoint * datapoints_per_block,
        1,
        1);
    const dim3 grid_size(
        DIVIDE_AND_CEIL(training_set.num_data_points, datapoints_per_block),
        1,
        1);

    //shared memory for matrix-vector partitial product and probability
    const size_t shared_memory_size = block_size.x * sizeof(FeatureType) 
         + datapoints_per_block * sizeof(FeatureType) * LABEL_CLASS ;
  

    // check that the resulting grid and block dimensions
    // dont' violate device limits
    if(checkDeviceProps(shared_memory_size, block_size, grid_size)) {

        // iterate if dimensions are okay
        for (size_t k = 0; k < training_options.num_epochs; k++) {

            // adjust step size with a modified version of simulated annealing
            annealed_step_size = training_options.config_params["initial_step_size"]
                                 / (1.0
                                    + (curr_num_epochs
                                       * training_set.num_data_points
                                       / characteristic_time));
            curr_num_epochs++;

            // call kernel and check for errors
            p_SgdWithSharedParameterVector
                <<<grid_size, block_size, shared_memory_size>>>(
                    d_data_points,
                    d_parameter_vector,
                    d_labels,
                    training_set.num_features,
                    training_set.num_data_points,
                    threads_per_datapoint,
                    threads_class_per_datapoint,
                    annealed_step_size);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());

        }
        // copy results from device memory back to host
        checkCudaErrors(cudaMemcpy(
            training_set.parameter_vector,
            d_parameter_vector,
            LABEL_CLASS * training_set.num_features * sizeof(FeatureType),
            cudaMemcpyDeviceToHost));

    }

    *training_options.step_size = annealed_step_size;

    cleanUp();
}