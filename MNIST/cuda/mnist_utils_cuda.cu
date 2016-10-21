#include <math.h>

#include "mnist_utils_cuda.cuh"
#include "utils/mnist_timer.h"



//------cublas function for softmax-------
void p_softmaxFunction(cublasHandle_t handle, 
    FeatureType* d_theta, 
    FeatureType* d_x_i,
    FeatureType* d_result,
    FeatureType* posibilities_positive,
    const size_t num_feats, 
    const size_t num_labels) {

    float alf = 1.0;
    float beta = 0;
    // refer to http://stackoverflow.com/questions/21164373/the-cublas-function-call-cublassgemv
    cublasSgemv(handle, CUBLAS_OP_T, num_feats, num_labels,
         &alf, d_theta, num_feats, d_x_i, 1, 
         &beta, d_result, 1);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(posibilities_positive, 
        d_result, 
        (LABEL_CLASS * sizeof(FeatureType)), 
        cudaMemcpyDeviceToHost));

    float sum = 0;

    for(size_t i=0 ; i< num_labels; i++){
        posibilities_positive[i] = exp(posibilities_positive[i]);
        sum += posibilities_positive[i];
    } 

    for(size_t i=0 ; i< num_labels; i++){
        posibilities_positive[i] /= sum;
    } 
}

// another implementation way
// slower than first one
// void p_softmaxFunction2(cublasHandle_t handle, 
//     FeatureType* d_theta, FeatureType* d_x_i,
//     FeatureType* posibilities_positive,
//     const size_t num_feats, 
//     const size_t num_labels){

//     for(size_t i=0;i< num_labels;i++){       

//         posibilities_positive[i] = p_dot_product(handle,
//             &d_theta[i*num_feats],
//             d_x_i,
//             num_feats);
//         // posibilities_positive[i] = a;
//     }

//     float sum = 0;

//     for(size_t i=0 ; i< num_labels; i++){
//         posibilities_positive[i] = exp(posibilities_positive[i]);
//         sum += posibilities_positive[i];
//     } 

//     for(size_t i=0 ; i< num_labels; i++){
//         posibilities_positive[i] /= sum;
//     } 

// }

// adds two device vectors with CuBLAS and stores the results in the first one
void p_add_vectors(cublasHandle_t handle, float* a, float* b, const size_t size, const float scale_for_a) {
    cublasSaxpy(handle, size, &scale_for_a, b, 1, a, 1);
}

// update parameter in parallel in CuBLAS
void p_updateParameters(cublasHandle_t handle, FeatureType* d_theta, FeatureType* d_gradient, size_t num_feats, float step_size, bool revert) {
    float sign = revert ? 1 : -1;
    step_size *= sign;
    cublasSaxpy(handle, num_feats, &step_size, d_gradient, 1, d_theta, 1);
}


float p_dot_product(cublasHandle_t handle, float* d_a, float* d_b, const size_t num_elems) {

    float result[1];
    cublasSdot (handle, num_elems, d_a, 1, d_b, 1, result);
    cudaDeviceSynchronize();
    return *result;
}




 

// initializes all values in array to a certain value
__device__ void d_memset(
    FeatureType* array,
    float value,
    size_t num_elements,
    size_t threads_per_mini_batch) {

    size_t tidx = threadIdx.x;
    for (size_t i = tidx; i < num_elements; i += threads_per_mini_batch) {
        array[i] = value;
    }
}




// update parametr
// thread is divide in several classes
// each number in one class calcualte strided
// vector add 
// __device__ void d_updateParametersForMiniBatch(
//     FeatureType* data_points,
//     FeatureType* probabilities_of_each,
//     FeatureType* parameter_vector,
//     size_t num_features,
//     size_t batch_size,
//     size_t threads_per_mini_batch,
//     size_t threads_class_per_datapoint) {

//     size_t tidx = threadIdx.x;
//     size_t bidx = blockIdx.x;

//     // index relative to each class of thread
//     size_t num_thread_each_class = threads_per_mini_batch / threads_class_per_datapoint;
//     size_t relative_tidx_each_class = tidx % num_thread_each_class;
//     size_t parameters_idx_each_class =  tidx / num_thread_each_class;
//     size_t num_parameter_each_class = LABEL_CLASS / threads_class_per_datapoint;
    
//     // for each loop update parameter in parallel by several class
//     // of threads
//     for(size_t m = 0; m < num_parameter_each_class; m++){

//         for (size_t i = relative_tidx_each_class; i < num_features; i += num_thread_each_class) {

//             float gradient_times_stepsize = 0;

//             size_t parameter_position = parameters_idx_each_class + m * threads_class_per_datapoint;

//             for (size_t j = 0; j < batch_size; j++) {
//                 // index of the point with respect to the whole dataset
//                 size_t point_idx = bidx * batch_size + j;
//                 // index of the feature with respect to all features in the dataset
//                 size_t feature_idx = point_idx * num_features + i;
//                 //gradient result 
//                 gradient_times_stepsize += data_points[feature_idx] 
//                     * probabilities_of_each[j*LABEL_CLASS + parameter_position];
//             }

//             atomicAdd(&parameter_vector[i + parameter_position * num_features], -gradient_times_stepsize);    
        
//         }
//     }    

// }



// general softmax function for all partitions
__device__ void d_softMaxFunction(
    FeatureType* posibility_each,
    size_t relative_tidx,
    size_t point_idx_in_block) {

    //calculate sum , each thread has a copy (++)
    float sum = 0;
    for (size_t i=0;i<LABEL_CLASS;i++){
        sum += posibility_each[point_idx_in_block * LABEL_CLASS + i];
    }
    __syncthreads();
    
    //calculate final posibility for each point
    if(relative_tidx < LABEL_CLASS){
        posibility_each[point_idx_in_block * LABEL_CLASS+relative_tidx] /= sum;
    }
    __syncthreads();
}






// verify the device properties satisfy the assumptions of the kernel
// check that the resulting grid and block dimensions
// dont' violate device limits
bool checkDeviceProps(
    size_t shared_memory_size,
    dim3 block_size,
    dim3 grid_size) {

    bool devicePropsOK = true;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (shared_memory_size > deviceProp.sharedMemPerBlock) {

        // printf("Shared Memory size exceeds maximum allowed size.\n");
        printf("SM-");
        devicePropsOK = false;
    }

    if (block_size.x > deviceProp.maxThreadsDim[0]
        || grid_size.x > deviceProp.maxGridSize[0]) {

        // printf("Grid or block size exceeds maximum allowed size.\n");
        printf("B-");
        devicePropsOK = false;
    }

    return devicePropsOK;
}

// updates the parameters (theta)
// void p_updateParameters(FeatureType* d_theta, FeatureType* d_gradient, size_t num_features, float step_size, bool revert) {
     // float sign = revert ? 1 : -1;
     // step_size *= sign;
     // addVectors(d_theta, d_gradient, num_features, step_size);
     // cublasSaxpy(handle, num_features, &step_size, d_gradient, 1, d_theta, 1);
//}
