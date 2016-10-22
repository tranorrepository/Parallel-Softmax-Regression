#define NUM_FEATURES      1024/16
#define NUM_SAMPLES       5000
#define NUM_TRAINING      4500
#define NUM_TESTING       500
#define STEP_SIZE         60000 //step size (eta)
#define NUM_EPOCHS        1
#define MAX_NUM_EPOCHS    1
#define DOUBLE_BUFFER_SIZE     500 
#define BUFFER_ITERATION  9
// #define BUFFER_ITERATION  NUM_TRAINING/DOUBLE_BUFFER_SIZE

typedef float FeatureType;
typedef float LabelType;
typedef float16 VectorFeatureType;

#define LOOP_PIPELINE __attribute__((xcl_pipeline_loop))
#define LOOP_UNROLL __attribute__((opencl_unroll_hint))
#define FADD_LAT 8



// dot product between two vectors
// use some techs to pipeine floating 
// point accumulation 
// please refer to http://www.xilinx.com/support/answers/62859.html
FeatureType cl_dotProduct(__local VectorFeatureType* a, __local VectorFeatureType* b, int size) {

    // result of partitial sum
    VectorFeatureType result_vector_p[FADD_LAT] __attribute__((xcl_array_partition(complete, 1)));
    // result vector
    VectorFeatureType result_vector = (0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f,
                                        0.0f,0.0f,0.0f,0.0f);
    FeatureType result = 0;

   // to pipeline floating point accumulation
    LOOP_UNROLL
     LOOP_INIT:for(int i = 0; i < FADD_LAT; i++) {    
        result_vector_p[i] = (0.0f,0.0f,0.0f,0.0f,
                                0.0f,0.0f,0.0f,0.0f,
                                0.0f,0.0f,0.0f,0.0f,
                                0.0f,0.0f,0.0f,0.0f);
    }

    // LOOP_PIPELINE
    LOOP_PIPELINE
    LOOPA:for(int i = 0; i < size; i += FADD_LAT){
    LOOP_UNROLL
    for (int j = 0; j < FADD_LAT; j++)
        result_vector_p[j] += a[j + i] * b[j + i];
    }


    
    LOOP_UNROLL
    LOOP_SUM_F: for (int k = 0; k < FADD_LAT; k++)
    {
        result_vector += result_vector_p[k];
    }



    // result sum
    result = result_vector.s0 + result_vector.s1 + result_vector.s2 + result_vector.s3
                + result_vector.s4 + result_vector.s5 + result_vector.s6 + result_vector.s7
                + result_vector.s8 + result_vector.s9 + result_vector.sa + result_vector.sb
                + result_vector.sc + result_vector.sd + result_vector.se + result_vector.sf;

    return result;
}

// hard logistic function
int cl_hardLogisticFunction(FeatureType exponent) {
    return (exponent >= 0) ? 1 : 0;
}

// logistic function
float cl_logisticFunction(FeatureType exponent) {
    return 1.0 / (1.0 + exp(-exponent));
}

// approximation of logistic function to avoid exponentiation
// float c_fauxLogisticFunction(FeatureType exponent) {

//     float e;

//     if (exponent < 0)
//         e = exponent * exponent + 1;
//     else
//         e = 1 / (exponent * exponent + 1);

//     return 1.0 / (1.0 + e);
// }

/*
* OpenCL kernal with single buffer caching
* input : datapoints labels and parameter
* output : parameter
*/
__attribute__ ((reqd_work_group_size(1, 1, 1)))
//__kernel void DigitRec(__global long long * global_training_set, __global long long * global_test_set, __global long long * global_results) {
__kernel void SgdLR(__global VectorFeatureType * global_data_points, 
    __global LabelType * global_labels, 
    __global VectorFeatureType * global_parameter_vector) {


    // datacopy event 
    event_t datacopy_evt[3];
    event_t databuffer_copy[BUFFER_ITERATION - 1];

    // Read data point from global memory
    __local VectorFeatureType parameter_vector[NUM_FEATURES];
  
    __local VectorFeatureType data_point[2][DOUBLE_BUFFER_SIZE * NUM_FEATURES];

    __local FeatureType labels[NUM_TRAINING];

    // Copy parameter and labels
    // and 500 samples into the first buffer
    datacopy_evt[0] = async_work_group_copy(parameter_vector, global_parameter_vector, NUM_FEATURES , 0);
    datacopy_evt[1] = async_work_group_copy(data_point[0], global_data_points, NUM_FEATURES * DOUBLE_BUFFER_SIZE , 0);
    datacopy_evt[2] = async_work_group_copy(labels, global_labels, NUM_TRAINING, 0);

    wait_group_events(3, datacopy_evt);

    // barrier(CLK_LOCAL_MEM_FENCE);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        // Iterate over all training instances (data points)
        // static int read = 0;
        // LOOP_PIPELINE
        // LOOP_UNROLL
        for(int buffer_iteration_number = 0; buffer_iteration_number < BUFFER_ITERATION; buffer_iteration_number++){
            // calculate execute and data copy buffer
            int buffer_execution_number = buffer_iteration_number % 2;
            int buffer_copy_number = (buffer_execution_number + 1) % 2;
            // copy data to another buffer
            if(buffer_iteration_number < (BUFFER_ITERATION - 1)){

                databuffer_copy[buffer_iteration_number] =  async_work_group_copy(data_point[buffer_copy_number], 
                                    &global_data_points[(buffer_iteration_number + 1) * DOUBLE_BUFFER_SIZE * NUM_FEATURES],
                                     NUM_FEATURES * DOUBLE_BUFFER_SIZE , 0);

            }
            // execute data in one buffer
            for (int i = 0; i < DOUBLE_BUFFER_SIZE; i++) {
                // calculate dot product
                FeatureType dot = cl_dotProduct(parameter_vector, &data_point[buffer_execution_number][i * NUM_FEATURES], NUM_FEATURES);
                // calculate probability
                float probability_of_positive = cl_hardLogisticFunction(dot);   
                //  calculate gradient
                float step = -(probability_of_positive - labels[i + buffer_iteration_number * DOUBLE_BUFFER_SIZE]) * STEP_SIZE;
                // fit into vector 16 data type
                VectorFeatureType step16 = (step, step, step, step,
                                                step, step, step, step,
                                                step, step, step, step,
                                                step, step, step, step);

                // finishes computation of (gradient * step size) and updates parameter vector
                LOOP_PIPELINE
                // __attribute__((opencl_unroll_hint(6)))
                for (int j = 0; j < NUM_FEATURES; j++){
                    // two ways same time
                    // parameter_vector[j].s0 += step * data_point[i * NUM_FEATURES + j].s0;
                    // parameter_vector[j].s1 += step * data_point[i * NUM_FEATURES + j].s1;
                    // parameter_vector[j].s2 += step * data_point[i * NUM_FEATURES + j].s2;
                    // parameter_vector[j].s3 += step * data_point[i * NUM_FEATURES + j].s3;
                    // parameter_vector[j].s4 += step * data_point[i * NUM_FEATURES + j].s4;
                    // parameter_vector[j].s5 += step * data_point[i * NUM_FEATURES + j].s5;
                    // parameter_vector[j].s6 += step * data_point[i * NUM_FEATURES + j].s6;
                    // parameter_vector[j].s7 += step * data_point[i * NUM_FEATURES + j].s7;
                    // parameter_vector[j].s8 += step * data_point[i * NUM_FEATURES + j].s8;
                    // parameter_vector[j].s9 += step * data_point[i * NUM_FEATURES + j].s9;
                    // parameter_vector[j].sa += step * data_point[i * NUM_FEATURES + j].sa;
                    // parameter_vector[j].sb += step * data_point[i * NUM_FEATURES + j].sb;
                    // parameter_vector[j].sc += step * data_point[i * NUM_FEATURES + j].sc;
                    // parameter_vector[j].sd += step * data_point[i * NUM_FEATURES + j].sd;
                    // parameter_vector[j].se += step * data_point[i * NUM_FEATURES + j].se;
                    // parameter_vector[j].sf += step * data_point[i * NUM_FEATURES + j].sf;
                    parameter_vector[j] += step16 * data_point[buffer_execution_number][i * NUM_FEATURES + j];
                }
            }

            if(buffer_iteration_number < (BUFFER_ITERATION - 1)){
                wait_group_events(1, &databuffer_copy[buffer_iteration_number]);
            }

        }
    }
    // barrier(CLK_LOCAL_MEM_FENCE);
    event_t result_evt = async_work_group_copy(global_parameter_vector, parameter_vector, NUM_FEATURES, 0);
    wait_group_events(1, &result_evt);
    // barrier(CLK_LOCAL_MEM_FENCE);
    // wait_group_events(1, &results_copy);
}

