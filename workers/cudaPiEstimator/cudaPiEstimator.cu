/*
 *  Experimental CUDA Pi estimator - based on mic version
 * Compiled with Cuda compiler.
 */

/**
 *  API Summary 5-2015
 *  Dont forget job_id -   job_id = job_handle;
 *  1. SendJobReceived() normally in constructor
 *  2. SendPercentDone()
 *  3. SendResults()
 */

#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; };
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
#endif

//#define DEBUG

#define CUDA

#include "cudaPiEstimator.hpp"
/*// Host side parallel support
#include <omp.h>*/

using namespace std;
using namespace boost::property_tree;
using namespace boost::posix_time;



/**
*  Kernels
**/

template< typename FLOAT >
__global__ void piKernel(FLOAT *reduction_sums, FLOAT step, unsigned long num_sims,
        long long unsigned integration_offset) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_sims) {
        float x = ((idx + integration_offset) - 0.5) * step;
        reduction_sums[idx] = 4.0 / (1.0 + x * x);
    }
}


/**
*
* Glue
*
*/

extern "C" bool HPCgearLaunch (HgTaskParams & task_params) {
    return HPCgearConnect<PiEstimatorWorker> (task_params);
}



/**
*
*  HgWorker Class implementation
*/

//  Constructor - reads JSON input and sets default values for this HgWorker
PiEstimatorWorker::PiEstimatorWorker (HgTaskParams & task_params) : HgWorker(task_params) {
    /**
    *  Get JSON values passed in
    *  and init WORKER_VALUES struct
    **/

    stringstream ss(task_params.get_source_data());
    ptree source_properties;
    read_json(ss, source_properties);

    workerValues.num_sims = source_properties.get<unsigned long long >("num_sims", 10000);
    workerValues.is_single = source_properties.get<bool>("single", false);

cout << "Send job received" << endl;
    SendJobReceived();


    /**
    *  Worker Bounds checking
    **/
    ostringstream error_message;
    bool is_error = false;





    if (is_error) {
        SendError(error_message.str());
        throw std::runtime_error(error_message.str());
    } 

}


void PiEstimatorWorker::operator()(ptree & results_properties) {

    begin_time = microsec_clock::local_time();

    // Get device properties  TODO:  Move down
    struct cudaDeviceProp     deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);


    const int MAX_SIMS_KERNEL = 50000000;
    long long unsigned sims_remaining = workerValues.num_sims;


    /*  Array for managing reductions  */
    double * reduction_sums;
    float  * reduction_sums_single;

    int mem_size;


    if (workerValues.is_single) {
        mem_size = MAX_SIMS_KERNEL * sizeof(float);
        cudaMallocManaged(&reduction_sums_single, mem_size);
    } else {
        mem_size = MAX_SIMS_KERNEL * sizeof(double);
        cudaMallocManaged(&reduction_sums, mem_size);
    }
    cout << "mem size, is_single " << mem_size << " " << workerValues.is_single << endl;


    double pi;
    double sum = 0.0;
    float pi_single;
    float sum_single = 0.0;

    double step        = 1.0 / static_cast<double>(workerValues.num_sims);
    float step_single  = 1.0 / static_cast<float>(workerValues.num_sims);

    long long unsigned integration_offset = 0L;

    while (sims_remaining > 0) {
        int sims_this_iteration = sims_remaining > MAX_SIMS_KERNEL ? MAX_SIMS_KERNEL : sims_remaining;

        cout << "sims_this_iteration " << sims_this_iteration << endl;

        if (workerValues.is_single) {
            memset (reduction_sums_single, 0, mem_size);
            getCudaOccupancyDetails(gridSize, &minGridSize, &blockSize, piKernel<float>, 0, sims_this_iteration);
            piKernel<<< gridSize, blockSize >>>(reduction_sums_single, step_single, sims_this_iteration, integration_offset);
        } else {
            memset (reduction_sums, 0, mem_size);
            getCudaOccupancyDetails(gridSize, &minGridSize, &blockSize, piKernel<double>, 0, sims_this_iteration);
            piKernel<<< gridSize, blockSize >>>(reduction_sums, step, sims_this_iteration, integration_offset);
        }
        getLastCudaError("Kernel execution failed");
        cudaDeviceSynchronize();



        //  Reduction on sums for this iteration
        for (int i = 0; i < sims_this_iteration; i++) {
            if (workerValues.is_single) {
                sum_single += reduction_sums_single[i];
            } else {
                sum += reduction_sums[i];
            }
        }

        SendPercentDone (1.0 - (static_cast<float>(sims_remaining) / workerValues.num_sims));

        sims_remaining -= sims_this_iteration;
        integration_offset += sims_this_iteration;

    }
    if (workerValues.is_single) {
        pi_single = step_single * sum_single;
    } else {
        pi = step * sum;
    }
    if (workerValues.is_single) {
        cudaFree(reduction_sums_single);
    } else {
        cudaFree(reduction_sums);
    }

    end_time = microsec_clock::local_time();
    elapsed_time = (time_duration (end_time - begin_time)).total_milliseconds();


    /**
    *
    *  Additional properties for results / JSON
    *  Refer to http://stackoverflow.com/questions/2114466/creating-json-arrays-in-boost-using-property-trees
    *
    */
    ptree pt_float_value;
    ptree pt_results_array;

    //  Do hardware details
    int num_cores = _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor) * deviceProperties.multiProcessorCount;
    ptree hardware_details;
    hardware_details.put("name", deviceProperties.name);
    hardware_details.put("cores", num_cores);
    hardware_details.put("clock", deviceProperties.clockRate * 1e-6f);
    hardware_details.put("streaming", deviceProperties.multiProcessorCount);

    //  All these will be pushed from server to browser via SSE
    results_properties.put("pi", workerValues.is_single ? pi_single : pi);
    results_properties.put("elapsed_time", elapsed_time);
    results_properties.add_child("hardware_details", hardware_details);


    ////  Also send results
    SendResults (results_properties);

}


