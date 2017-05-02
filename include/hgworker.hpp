// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <algorithm>

#ifdef CUDA
// CUDA runtime & vector types
#include <cuda_runtime.h>
#include <vector_types.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

//  JSON and other library Support
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>

// ZeroMQ
#include <zmq.hpp>

//  Gravity Neutral Launch/Task Parameters
#include "hglaunch.hpp"


/**
*  Shared wrapper for CUDA configurator template
**/
#ifdef CUDA
template<class T>
__inline__ __host__ CUDART_DEVICE cudaError_t getCudaOccupancyDetails(
        int    & gridSize,
        int    *minGridSize,
        int    *blockSize,
        T       func,
        size_t  dynamicSMemSize = 0,
        int     blockSizeLimit = 0) {
    cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, 0, blockSizeLimit);
    // Round up according to array size
    gridSize = (blockSizeLimit + *blockSize - 1) / *blockSize;
#ifdef DEBUG
    cout << "gridSize is " << gridSize << " blockSize is " << *blockSize << " minGridSize " << *minGridSize << endl;
#endif
//  Only to get rid of compiler warning
    return (cudaError_t) 0;
}
#endif

/**
*  HgWorker
*  Abstract base class for Gravity Connect workers
*/

class HgWorker {
public:
  //  HgWorker(const char * source_data, const char * host);
    HgWorker(HgTaskParams & task_params);
    virtual ~HgWorker();
    //  Waits for message from server / SSE that it's ready for messages from us
protected:
    //  Common elements for all HgWorkers
    float elapsed_time;
    boost::property_tree::ptree pt_elapsed_time;
    // Generalized time
    boost::posix_time::ptime begin_time, end_time;
    //  Gearman: gearman_job_handle(job)
  //  std::string job_id;
    #ifdef CUDA
    //  Kernel parameters
    int blockSize;      // Block size returned from the launch configurator
    int minGridSize;    // Minimum grid size to achieve maximum occupancy for full device launch
    int gridSize;       // Actual grid size needed, based on input size
    //  CUDA kernal timers
    cudaEvent_t start;
    cudaEvent_t stop;
    #endif
    void SendPtree(boost::property_tree::ptree & ptree);
    bool SendJobReceived(void);
    bool SendError(const std::string & error_message);
    bool SendPercentDone(float percent);
    bool SendResults(boost::property_tree::ptree & results_properties);

private:
    bool use_zmq;
    zmq::context_t * context ;
    zmq::socket_t * send_socket;
    zmq::socket_t * send_socket2;

    std::string port;
    // cant this also come from in from the web server??
    // or from the gearman caller??

};

HgWorker::HgWorker(HgTaskParams & task_params) : context(0), send_socket(0), send_socket2(0), port(""), use_zmq(true) {

    //  Read some properties from the JSON data
    std::stringstream ss(task_params.get_source_data());
    boost::property_tree::ptree source_properties;
    boost::property_tree::read_json(ss, source_properties);

    if (! source_properties.get<bool>("use_zmq", true) || task_params.get_job_handle() == std::string("cli_task")) {
        std::cout << "Setting use_zmq false " << std::endl;
        use_zmq = false;
    }

    std::cout << "use_zmq is " << use_zmq << std::endl;
    if (use_zmq) {

//        std::string port = source_properties.get<std::string>("port", "");
        port = source_properties.get<std::string>("port", "");

        //  ZMQ setup
        context = new zmq::context_t();
        send_socket = new zmq::socket_t(*context, ZMQ_PUSH);
        std::string connect_str =
                std::string("tcp://") + std::string(task_params.get_host()) + std::string(":") + port;

        // TEMP WORKAROUND UNTIL gearman OFF quantum
  ////////      connect_str = std::string("tcp://") + std::string("192.168.1.26") + std::string(":") + port2;



        std::cout << "Worker connecting to " << connect_str << std::endl;
        send_socket->connect(connect_str.c_str());
        std::cout << "Connecing done" << std::endl;

        // DEBUGGING ONLY
/*
            int port_num  = boost::lexical_cast<int>(port);
           port_num++;
            std::cout << "lex casting port_num " << port_num << " to string" << std::endl;

            std::string port2 = boost::lexical_cast<std::string>(port_num);

        send_socket2 = new zmq::socket_t(*context, ZMQ_PUSH);
//  26 will eventually be get_host()
        connect_str = std::string("tcp://") + std::string("192.168.1.26") + std::string(":") + port2;
            std::cout << "2nd socket " << connect_str << std::endl;
            send_socket2->connect(connect_str.c_str());*/



    }
}


HgWorker::~HgWorker() {
    std::cout << "GN Destructor" << std::endl;
    if (send_socket) {
        delete send_socket;
    }
    if (send_socket2) {
        delete send_socket2;
    }
    if (context) {
        delete context;
    }
    std::cout << "EXIT GN Destructor - zmq context deleted " << std::endl;
}

//  Todo:  Return error if socket send fails?
void HgWorker::SendPtree(boost::property_tree::ptree & ptree) {
    if (use_zmq) {
        // Convert to string
        std::ostringstream string_stream;
        boost::property_tree::write_json(string_stream, ptree);
        std::string message = string_stream.str();

        send_socket->send(&message[0], message.length());

        // DEBUGGING ONLY
        if (send_socket2) {
            std::cout << "Sending a msg on socket2" << std::endl;
            send_socket2->send(&message[0], message.length());
        }
        //    std::cout << "Sent  ptree " << message << std::endl;
    }
}

/**
 *  Job received successfully - Return JSON object
 */
bool HgWorker::SendJobReceived(void) {
    boost::property_tree::ptree ptree;
    ptree.put("type", "received");
    ptree.put("data", "Bogus data");
    SendPtree (ptree);
    return true;
}

/**
 *  Processing error occurred - Return JSON object
 */
bool HgWorker::SendError(const std::string & error_message) {

    std::cout << "Calling SendError()  " << std::endl;

    // Format message as JSON
    boost::property_tree::ptree ptree;
    ptree.put("type", "error");
    ptree.put("data", error_message);
    SendPtree (ptree);
    return true;
}

/**
 *  Percent complete - Return JSON object
 */
bool HgWorker::SendPercentDone(float percent) {
    percent = std::min (percent, 1.0f);
    boost::property_tree::ptree ptree;
    ptree.put("type", "progress");
    ptree.put("data", percent);
    SendPtree (ptree);
    return true;
}

/**
 *  Final results as sent from worker - Return JSON object
 */
bool HgWorker::SendResults(boost::property_tree::ptree & results_properties) {
    boost::property_tree::ptree ptree;
    ptree.put("type", "finish");
    ptree.add_child("data", results_properties);
    SendPtree (ptree);
    return true;
}



/**
*
*  gravityConnect: Callable glue between Modules that inherit from HgWorker class <--> Gearman C API
*/
template<class Worker> bool gravityConnect (HgTaskParams & task_params) {

    //  TODO:  How are we using source_len?  Is it needed?
    //  For JSON / property trees
    boost::property_tree::ptree results_properties;

    try {
        Worker HgWorker(task_params);
        HgWorker(results_properties);
    }
    catch(std::exception &e) {
        std::cout << e.what() << std::endl;
        results_properties.put("error", e.what());
        results_properties.put("elapsed_time", 0);
    }

    write_json(*task_params.results_stream, results_properties);

    return true;
}