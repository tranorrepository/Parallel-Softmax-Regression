#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"

#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

/*
 * Parallel approach to Stochastic Gradient Descent #3 - Vivado HLS - C:
 *
 */

// needs this when the kernel is in a cpp file
// extern "C" {

// dot product between two vectors
FeatureType c_dotProduct(FeatureType* a, FeatureType* b, size_t size) {

    FeatureType result = 0;

    size_t j;
    for (j = 0; j < size; j++)
        result += a[j] * b[j];

    return result;
}

// hard logistic function
int c_hardLogisticFunction(FeatureType exponent) {
    return (exponent >= 0) ? 1 : 0;
}

// logistic function
float c_logisticFunction(FeatureType exponent) {
    return 1.0 / (1.0 + exp(-exponent));
}

// approximation of logistic function to avoid exponentiation
float c_fauxLogisticFunction(FeatureType exponent) {

    float e;

    if (exponent < 0)
        e = exponent * exponent + 1;
    else
        e = 1 / (exponent * exponent + 1);

    return 1.0 / (1.0 + e);
}

// Top-level Kernel
// Stochastic Gradient Descent for Logistic Regression
void SgdLR(
    FeatureType* global_data_points,
    LabelType* global_labels,
    FeatureType* global_parameter_vector) {

    DO_PRAGMA(HLS INTERFACE ap_bus port=global_data_points offset=slave bundle=gmem depth=DATA_SET_SIZE)
    DO_PRAGMA(HLS INTERFACE ap_bus port=global_labels offset=slave bundle=gmem depth=NUM_SAMPLES)
    DO_PRAGMA(HLS INTERFACE ap_bus port=global_parameter_vector offset=slave bundle=gmem depth=NUM_FEATURES)
    // DO_PRAGMA(HLS INTERFACE m_axi port=global_data_points offset=slave bundle=gmem depth=DATA_SET_SIZE)
    // DO_PRAGMA(HLS INTERFACE m_axi port=global_labels offset=slave bundle=gmem depth=NUM_SAMPLES)
    // DO_PRAGMA(HLS INTERFACE m_axi port=global_parameter_vector offset=slave bundle=gmem depth=NUM_FEATURES)
    // DO_PRAGMA(HLS INTERFACE s_axilite port=global_data_points bundle=control)
    // DO_PRAGMA(HLS INTERFACE s_axilite port=global_labels bundle=control)
    // DO_PRAGMA(HLS INTERFACE s_axilite port=global_parameter_vector bundle=control)
    // DO_PRAGMA(HLS INTERFACE s_axilite port=return bundle=control)

    printf("Entered kernel\n"); // for debugging

    // Device vectors
    FeatureType data_point_i[NUM_FEATURES];
    FeatureType parameter_vector[NUM_FEATURES];

    // Auxiliar variables for cycles' indexes
    size_t i;
    size_t j;
    size_t epoch;

    // Read parameter vector from global memory
    for (j = 0; j < NUM_FEATURES; j++)
        parameter_vector[j] = global_parameter_vector[j];

    // Auxiliar pointer to go through the data set
    // regular indexing in the form "stride * outer_idx + inner_idx" caused an error
    FeatureType* ptr_to_global_data_points = &global_data_points[0];

    for (epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        // reset pointer to first point in the dataset
        ptr_to_global_data_points = &global_data_points[0];

        // Iterate over all training instances (data points)
        static int read = 0;
        for (i = 0; i < NUM_TRAINING; i++) {
            // Read data point from global memory
            read = 0;
            // printf("\n\n");
            for (j = 0; j < NUM_FEATURES; j++)
                data_point_i[j] = ptr_to_global_data_points[j];

            // starts computation of gradient
            FeatureType dot = c_dotProduct(parameter_vector, data_point_i, NUM_FEATURES);

            float probability_of_positive = c_hardLogisticFunction(dot);

            // printf("\npoint %d --> dot:%f | pps:%f | label:%f", i, dot, probability_of_positive, global_labels[i]);

            float step = -(probability_of_positive - global_labels[i]) * STEP_SIZE;

            // finishes computation of (gradient * step size) and updates parameter vector
            for (j = 0; j < NUM_FEATURES; j++) {
                parameter_vector[j] += step * data_point_i[j];
            }


            // advance to next data point
            ptr_to_global_data_points = &ptr_to_global_data_points[NUM_FEATURES];
        }
    }

    // copy results back to global memory
    for (j = 0; j < NUM_FEATURES; j++) {
        global_parameter_vector[j] = parameter_vector[j];
    }

    printf("Exited kernel\n"); // for debugging

    return;
}


// } // end of extern C
