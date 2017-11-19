//
//  CLKernel.cpp
//  FFT
//
//  Created by Tim on 10/23/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//

#include <fstream>
#include <stdlib.h>

#include "CLKernel.h"

CLKernel::CLKernel(const char *filename, const char* kernel_name, cl_device_type device_type) {
    this->device_type = device_type;
    this->kernel_name = kernel_name;
    size_t code_size = (size_t ) load_file_to_memory(filename);
    cl_int create_binary_status;

    // Connect to a compute device
    //
    err = clGetDeviceIDs(NULL, this->device_type, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        exit(EXIT_FAILURE);
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        exit(EXIT_FAILURE);
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command queue!\n");
        exit(EXIT_FAILURE);
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *) &code_size, (const unsigned char **) & kernel_code, &create_binary_status, &err);
    if (!program)
    {
        printf("Error = %d\n", err);
        printf("Error: Failed to create compute program!\n");
        exit(EXIT_FAILURE);
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(EXIT_FAILURE);
    }

}

int CLKernel::load_to_memory() {
    // For now, the user must put something in memobj_array before calling this function
    load_data();

    const unsigned int num_of_mem = memobj_array.size();

    clmem_array.reserve(num_of_mem);

    for(int i = 0; i < num_of_mem; i++) {
        // Temp variables
        int elt_size = memobj_array[i].get_element_size();
        int length = memobj_array[i].get_length();
        void * mem_data = memobj_array[i].get_data();
        cl_mem_flags flags = memobj_array[i].get_flags();

        // Create the cl_mem object/handle
        clmem_array[i] = clCreateBuffer(context, flags, elt_size * length,
                                        NULL, NULL);
        if (!clmem_array[i])
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(EXIT_FAILURE);
        }
        // Queue up the opertion to copy the data to the device
        err = clEnqueueWriteBuffer(commands, clmem_array[i], CL_TRUE, 0,
                                   elt_size * length, mem_data, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(EXIT_FAILURE);
        }
        // Set the cl_mem buffer as the respective kernel argument
        err  = clSetKernelArg(kernel, i, sizeof(clmem_array[i]), &clmem_array[i]);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(EXIT_FAILURE);
        }
    }
    return EXIT_SUCCESS;
}

std::vector<CLMemObj> CLKernel::get_memory_objs() {
    return memobj_array;
}

// This function can probably be improved by somehow connecting it with
// CLMemObj, because elt_size and len are already part of CLMemObj
void CLKernel::read_clmem(int idx, int elt_size, int len, void * out_data) {
    err = clEnqueueReadBuffer(commands, clmem_array[idx], CL_TRUE, 0,
                              elt_size * len, out_data, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}

int CLKernel::run_kernel() {
    setup_kernel();
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    cl_event e;

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &e);
    if (err)
    {
        printf("Error: Failed to execute kernel! Error number: %d\n", err);
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &e);
    return EXIT_SUCCESS;
}

void CLKernel::finish_and_clean_up() {
    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Shutdown and cleanup
    const unsigned int num_of_mem = clmem_array.size();
    for(int i = 0; i < num_of_mem; i++) {
        clReleaseMemObject(clmem_array[i]);
    }
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    clean_up(); // Clean up any child class stuff
}

int CLKernel::load_file_to_memory(const char *filename) {
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
    {
        kernel_code = NULL;
        return -1; // -1 means file opening fail
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    kernel_code = (char *)malloc(size+1);
    if ((unsigned int) size != fread(kernel_code, sizeof(char), size, f))
    {
        free(kernel_code);
        return -2; // -2 means file reading fail
    }
    fclose(f);
    (kernel_code)[size] = 0;
    return size;
}

void CLKernel::set_global(int global_work_size) {
  global = global_work_size;
}

void CLKernel::set_local(int local_work_size) {
  local = local_work_size;
}
