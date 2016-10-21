//
//  CLKernel.h
//  FFT
//
//  Created by Tim on 10/23/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//

#ifndef __CLKernel__Harness__
#define __CLKernel__Harness__

#include <stdio.h>
#include <vector>
#include <CL/cl.h>

#include "CLMemObj.h"

enum type { READ, WRITE };

class CLKernel {
public:
    // Constructor; sets up/initializes device, context, program, memobj_array
    CLKernel(const char *, const char*, cl_device_type);
    // Public functions and virtual functions
    std::vector<CLMemObj> get_memory_objs();
    void set_global(int global_work_size);
    void set_local(int local_work_size);
    int run_kernel();                 // run the kernel
    int load_to_memory();             // set up cl_mem buffers using memobj_array
    void finish_and_clean_up();
    virtual int setup_kernel() = 0;   // setup anything needed before running kernel
    virtual int load_data() = 0;
    virtual bool check_results() = 0;
    virtual void clean_up() = 0;
protected:
    // Protected variables
    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation
    int err;                          // error code returned from api calls
    const char *kernel_name;          // name of kernel
    std::vector<CLMemObj> memobj_array;          // array of memory objects
    std::vector<cl_mem> clmem_array;             // array of cl_mem buffers
    // Protected functions
    void read_clmem(int, int, int, void *);  // get the clmem at index
    void clean_up_mem_objs();
private:
    // Private variables
    char *kernel_code;                // the kernel code
    cl_device_type device_type;       // compute device type
    cl_device_id device_id;           // compute device id
    cl_context context;               // compute context
    cl_command_queue commands;        // compute command queue
    cl_program program;               // compute program
    cl_kernel kernel;                 // compute kernel
    // Private functions
    int load_file_to_memory(const char *filename);
};

#endif /* defined(__CLKernel__Harness__) */
