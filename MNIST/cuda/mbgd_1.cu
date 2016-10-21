#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "mbgd_1.h"
#include "mnist_utils_cuda.cuh"

/* Parallel approach to batch gradient descent using multiple "mini-batches"
rather than the whole batch. Each block is assigned to one mini-batch with
multiple threads assigned to each data point. */


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

    checkCudaErrors(cudaMalloc(&d_parameter_vector,LABEL_CLASS * num_features 
                                * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(&d_data_points, num_data_points * num_features 
                                * sizeof(FeatureType)));
    checkCudaErrors(cudaMalloc(&d_labels, num_data_points * sizeof(LabelType)));

    checkCudaErrors(cudaMemcpy(d_data_points, data_points, num_data_points
                                * num_features * sizeof(FeatureType),
                                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_labels, labels, num_data_points
                                * sizeof(LabelType), 
                                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_parameter_vector,parameter_vector,
                                LABEL_CLASS * num_features * sizeof(FeatureType), 
                                    cudaMemcpyHostToDevice));
}


static void cleanUp() {

    checkCudaErrors(cudaFree(d_parameter_vector));
    checkCudaErrors(cudaFree(d_data_points));
    checkCudaErrors(cudaFree(d_labels));
}


// update parametr
// thread is divide in several classes
// each number in one class calcualte strided
// vecor add
static __device__ void d_updateParametersForMiniBatch(
    FeatureType* data_points,
    FeatureType* probabilities_of_each,
    FeatureType* parameter_vector,
    size_t num_features,
    size_t batch_size,
    size_t threads_per_mini_batch,
    size_t threads_class_per_datapoint) {

    size_t tidx = threadIdx.x;
    size_t bidx = blockIdx.x;

    // index relative to each class of thread
    size_t num_thread_each_class = threads_per_mini_batch / threads_class_per_datapoint;
    size_t relative_tidx_each_class = tidx % num_thread_each_class;
    size_t parameters_idx_each_class =  tidx / num_thread_each_class;
    size_t num_parameter_each_class = LABEL_CLASS / threads_class_per_datapoint;
    

    // for each loop update parameter in parallel by several class
    // of threads
    for(size_t m = 0; m < num_parameter_each_class; m++){

        for (size_t i = relative_tidx_each_class; i < num_features; i += num_thread_each_class) {

            float gradient_times_stepsize = 0;

            size_t parameter_position = parameters_idx_each_class + m * threads_class_per_datapoint;

            for (size_t j = 0; j < batch_size; j++) {
                // index of the point with respect to the whole dataset
                // size_t point_idx = bidx * batch_size + j;
                // index of the feature with respect to all features in the dataset
                size_t feature_idx = j * num_features + i;
                //gradient result 
                gradient_times_stepsize += data_points[feature_idx] 
                    * probabilities_of_each[j*LABEL_CLASS + parameter_position];
            }

            atomicAdd(&parameter_vector[i + parameter_position * num_features], -gradient_times_stepsize);    
        
        }
    }    

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

    // index relative to each class of thread
    size_t num_thread_each_class = threads_per_datapoint / threads_class_per_datapoint;
    size_t relative_tidx_each_class = thread_offset % num_thread_each_class;
    size_t parameters_idx_each_class =  thread_offset / num_thread_each_class;

    // calculate parameter in parallel by several class of threads
    for (size_t j = relative_tidx_each_class; j < num_features; j += num_thread_each_class)
        partial_dot += data_point_i[j] * parameter_vector[j + parameters_idx_each_class * num_features];

    // result of the partial dot product is stored in shared memory
    shared_memory[threadIdx.x] = partial_dot;
}


static __global__ void p_MiniBatchGradientDescent(
    FeatureType* data_points,
    FeatureType* parameter_vector,
    LabelType* labels,
    size_t num_features,
    size_t num_data_points,
    size_t batch_size,
    size_t threads_per_datapoint,
    size_t threads_class_per_datapoint,
    double step_size) {

    extern __shared__ FeatureType shared_memory[];
    // FeatureType *gradient = shared_memory;
    size_t threads_per_mini_batch = threads_per_datapoint * batch_size;
    
    // memory for matrix-vector product
    float *dot_product = shared_memory;

    // memory for possibility
    float *probabilities_of_each = (float*)&dot_product[threads_per_mini_batch];

    // shared memory space for cache datapoints
    float *shared_data_points = (float*)&probabilities_of_each[batch_size 
                            * LABEL_CLASS]; 
    
    size_t tidx = threadIdx.x;
    size_t num_parameter_each_class = LABEL_CLASS / threads_class_per_datapoint;

    //index reletive to all datapoint
    size_t point_idx = (blockIdx.x * batch_size)
                     + (tidx / threads_per_datapoint);
    // index relative to the datapoint instead of the block
    size_t relative_tidx = tidx % threads_per_datapoint;
    size_t point_idx_in_shmem = tidx - relative_tidx;
    size_t point_idx_in_block = tidx / threads_per_datapoint;
    // index relative to each class of thread
    size_t num_thread_each_class = threads_per_datapoint / threads_class_per_datapoint;
    size_t relative_tidx_each_class = relative_tidx % num_thread_each_class;
    // size_t parameters_idx_each_class =  relative_tidx / num_thread_each_class;
    
    if (point_idx < num_data_points) {

        // possibilly copy the data to the shared memory first...
        for (size_t j = relative_tidx; j < num_features; j += threads_per_datapoint){
            shared_data_points[j + point_idx_in_block * num_features]
                =  data_points[point_idx * num_features + j];
        }

        // __syncthreads();

        // compute partial matrix-vector product
        for(size_t i = 0; i < num_parameter_each_class; i++){
            
           
            d_partialMatrixVectorProduct(
                &shared_data_points[point_idx_in_block * num_features],
                &parameter_vector[i * threads_class_per_datapoint * num_features],
                dot_product,
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
        

        float reduced_stepsize = step_size / batch_size;
        
        //calculate step_size_times_prob_i_minus_label_i, store in the same position
        //calculate eta * {y(i)=k}−P(y(i)=k|x(i)
        if(relative_tidx < LABEL_CLASS){
            if(labels[point_idx]==relative_tidx){
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] -= 1;
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= reduced_stepsize;
            }else{                   
                probabilities_of_each[point_idx_in_block * LABEL_CLASS+relative_tidx] *= reduced_stepsize;
            }
        }
        
        __syncthreads();
       
        // update parameter
       d_updateParametersForMiniBatch(
            shared_data_points,
            probabilities_of_each,
            parameter_vector,
            num_features,
            batch_size,
            threads_per_mini_batch,
            threads_class_per_datapoint);
     
    }   
}


void trainParallelMiniBatchGradientDescent( 
    DataSet training_set,
    TrainingOptions training_options ) {

    // shuffle data points
    /* shuffleKeyValue( training_set.data_points, training_set.labels, 
                     training_set.num_data_points, training_set.num_features ); */

    // allocae memory and move to GPU
    setCudaVariables( training_set.num_features,
                      training_set.num_data_points,
                      training_set.data_points, 
                      training_set.labels, 
                      training_set.parameter_vector );
    
    // read configuration parameters and initialize grid and block dimensions
    double step_size = *training_options.step_size;

    const double threads_per_datapoint =
            (fieldExists(training_options.config_params, "threads_per_datapoint"))
            ? training_options.config_params["threads_per_datapoint"]
            : THREADS_PER_DATAPOINT;

    const double batch_size =
            (fieldExists(training_options.config_params, "batch_size"))
            ? training_options.config_params["batch_size"]
            : BATCH_SIZE;

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

    const dim3 block_size(threads_per_datapoint * batch_size, 1, 1);
    size_t num_blocks = DIVIDE_AND_CEIL( training_set.num_data_points,
                                            batch_size );
    const dim3 grid_size(num_blocks, 1, 1);

    FeatureType threads_per_batch = threads_per_datapoint * batch_size;
    //shared Memory for posibility,dot product and gradient
    const size_t shared_memory_size = LABEL_CLASS * batch_size * sizeof(float) 
            + (threads_per_batch) * sizeof(FeatureType)
            + batch_size * training_set.num_features * sizeof(FeatureType);

 
    if (checkDeviceProps(shared_memory_size, block_size, grid_size)) {
        // iterate if dimensions are okay
        for (size_t k = 0; k < training_options.num_epochs; k++) {
            annealed_step_size = training_options.config_params["initial_step_size"]
                                / (1.0
                                    + (curr_num_epochs
                                       * training_set.num_data_points
                                       / characteristic_time));
            curr_num_epochs++;

            // adjust step size with a modified version of simulated annealing

            // call kernel and check for errors
            p_MiniBatchGradientDescent
                    <<<grid_size, block_size, shared_memory_size>>>(
                            d_data_points, 
                            d_parameter_vector,
                            d_labels,
                            training_set.num_features, 
                            training_set.num_data_points, 
                            batch_size,
                            threads_per_datapoint,
                            threads_class_per_datapoint,
                            annealed_step_size );
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaMemcpy( training_set.parameter_vector, 
                                    d_parameter_vector,
                                    LABEL_CLASS * training_set.num_features 
                                    * sizeof(FeatureType), 
                                    cudaMemcpyDeviceToHost));
    }

    *training_options.step_size = annealed_step_size; 

    cleanUp();
}
