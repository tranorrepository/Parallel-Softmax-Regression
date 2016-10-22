# Parallelized Softmax Regression for digit classification #

#### tl;dr: ####

Read the [note](#paths) about paths and then go to [Running the code](#run).

## Reference: ##

This project is to speed up the training process of softmax regression, the file name and classes in this project is the name
as mnist.

## Description: ##

This project contains several different versions of the *Gradient Descent* method used to train a *softmax regression* model. This model can be used for classification problems, including digit recognition. The contents of the different modules are explained below:

<a name="paths"></a>**Note:** some libraries (OpenBLAS, CuBLAS, CUDA, Vivado HLS, ...) may be installed in different directories in your system, and thus may require the modification of paths in the Makefiles, setup `.sh` scripts (or the `LD_LIBRARY_PATH` environment variable) and `.tcl` scripts.

### cpp ###

This folder contains the baselines (C++ serial implementation) for **Batch Gradient Descent** **Stochastic Gradient Descent** and **MiniBatch Gradient Descent**. The use of OpenBLAS for linear algebra procedures reduces the execution time of the baselines considerably. In order to use OpenBLAS, set the path to your installed BLAS library (see the [note](#paths) about paths) and make sure the Makefile contains the flag `-DUSE_OPENBLAS`.

### cuda ###

This folder contains four variations of **Stochastic Gradient Descent**. The first one, `sgd_cublas`, uses calls to CuBLAS functions for linear algebra operations. This version is very slow, so it was removed from the default Makefile. If you want to test it, add it to the `cuda:` target and use `CUBLAS_INCLUDEANDLINK`. The second version, `sgd_single_point`, uses both data and model parallelism, and it runs faster than the CPU versions under the right settings. It has a custom implementation of the linear algebra procedures. The third version, which is a minibatch-1 version, speeds up a lot by updating parameter by
minibatch. The last version, which is minibatch-2, choose a different way to parallize threads with regard to the first one. 

### sdaccel-c ###

This folder contains a third alternative version of **Stochastic Gradient Descent** that uses the SDAccel harness. Due to the conventions of the harness, this folder has a very different structure from the two above. It has its own Makefile inside along with `.tcl` scripts that make the *solution*.

## <a name="run"></a>Running the code: ###

First, make sure the paths in the setup script in the scripts folder are correct. Then, source the script:

```bash
. scripts/setup.sh
```

If you are working on the Zhang servers, also source the setup scripts located at the top level directory of the reconfigurable-benchmark repo:

```bash
. ../setup-zhang-01.sh
```

After that, adjust the paths in the Makefiles (and `.tcl` scripts).

Then,

  * In order to test the **baselines** alone, use the following commands in a terminal standing at the current directory:
```bash
make clean
make baseline
bin/baseline [options]
```
Or just
```bash
. compile_and_run_baseline.sh [options]
```
For the different configuration options, see `utils/mnist_utils.cpp`.

Make sure the data (features and labels) are located in `./data/`.

  * In order to run the **CUDA** versions along with the baselines, run:
```bash
make clean
make cuda
bin/cuda [options]
```
Or just
```bash
. compile_and_all.sh [options]
```
  * In order to run the emulations for the SDAccel versions, follow the tutorial in the Digit-Recognition project.
