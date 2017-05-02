/**
 *  gearWorker.cpp
 *  Generic worker loader
 *  Loads a worker at entry point:
 *  launchGNWorker(&source_string[0], workload_size, results_stream);
 *  Supports both CUDA and Mic workers via this entry point
 **/


//#include "gear_config.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <signal.h>

#include <libgearman/gearman.h>
#include <boost/program_options.hpp>
#include <hglaunch.hpp>

using namespace std;
using namespace boost::program_options;

/**
*  Utilities so build can say -DWORKERNAME=worker_name on compile
*/
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)



struct worker_options_t {
    bool chunk;
    bool status;
    bool unique;
    bool verbose;
    std::string host;

    worker_options_t():
            chunk(false),
            status(false),
            unique(false),
            verbose(true),
            host ("localhost") { }
};

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

static gearman_return_t gear_worker (gearman_job_st * job, void * context) {
    worker_options_t options = *((worker_options_t *) context);

    const char *workload = (const char *) gearman_job_workload(job);
    const size_t workload_size = gearman_job_workload_size(job);

    //  Kludge fix for nodeJS workloads. Seems to not be null terminated
    string workload_string(workload, workload_size);
    //workload_string.resize(workload_size);

    if (options.verbose) {
        cout << "gearWorker Received " << workload_string << " " << workload_size << " bytes" << endl;
    }
    cout << "gearWorker - worker dispatch.." <<  "host is " <<  options.host << endl;

/*    string temp_string(workload);
    string source_string = temp_string.substr(0, workload_size);
    string job_handle(gearman_job_handle(job));

    cout << "Workload is " << source_string << " Job handle " << job_handle << endl;*/

    ostringstream results_stream;
//    HPCgearLaunch (&source_string[0], workload_size, &job_handle[0], results_stream);
 //   HPCgearLaunch (&source_string[0], workload_size, &options->host[0], results_stream);
    HgTaskParams task_params (workload_string.c_str(), workload_size, gearman_job_handle(job), options.host.c_str(),
                              &results_stream);
/*
    task_params.source_data     = workload;
    task_params.source_len      = workload_size;
    task_params.job_handle      = gearman_job_handle(job);
    task_params.host            = options->host.c_str();
    task_params.results_stream  = results_stream;*/

    HPCgearLaunch (task_params);


    if (options.status) {
        // Notice that we send based on y divided by zero.
        cout << "Sending a STATUS" << endl;
        if (gearman_failed(gearman_job_send_status(job, (uint32_t)0, (uint32_t)workload_size))) {
            return GEARMAN_ERROR;
        }
    }

//    if (gearman_failed(gearman_job_send_data(job, &result[0], workload_size))) {
    //  Send the results as a string
 //   string results = results_stream.str();
   //     if (gearman_failed(gearman_job_send_data(job, &results[0], results.length()))) {
          if (gearman_failed(gearman_job_send_data(job, results_stream.str().c_str(), results_stream.str().length()))) {
        return GEARMAN_ERROR;
    }

/*    if (options.status) {
        // Notice that we send based on y divided by zero.
        if (gearman_failed(gearman_job_send_status(job, (uint32_t)workload_size, (uint32_t)workload_size))) {
            return GEARMAN_ERROR;
        }
    }*/

    if (options.verbose) {
       // cout << "Job=" << gearman_job_handle(job);
    }
    if (options.unique and options.verbose) {
        cout << "Unique=" << gearman_job_unique(job);
    }
    if (options.verbose) {
 /*       std::cout << "  Reversed=";
        std::cout.write(results_stream, dest_size);
        std::cout << std::endl;*/
      //  cout << results_stream << endl;
    }
    cout << endl << "Returning SUCCESS.."  << endl;

    return GEARMAN_SUCCESS;

}


int main(int args, char *argv[]) {
    uint64_t limit;
    worker_options_t options;
    int timeout;

    in_port_t port;
    string host;
    string identifier;
    options_description desc("Options");
    desc.add_options()
            ("help", "Options related to the program.")
            ("host,h", boost::program_options::value<std::string>(&host)->default_value("localhost"),"Connect to the host")
            ("identifier", boost::program_options::value<std::string>(&identifier),"Assign identifier")
            ("port,p", boost::program_options::value<in_port_t>(&port)->default_value(GEARMAN_DEFAULT_TCP_PORT), "Port number use for connection")
            ("count,c", boost::program_options::value<uint64_t>(&limit)->default_value(0), "Number of jobs to run before exiting")
            ("timeout,u", boost::program_options::value<int>(&timeout)->default_value(-1), "Timeout in milliseconds")
            ("chunk,d", boost::program_options::bool_switch(&options.chunk)->default_value(false), "Send result back in data chunks")
            ("status,s", boost::program_options::bool_switch(&options.status)->default_value(false), "Send status updates and sleep while running job")
            ("unique,u", boost::program_options::bool_switch(&options.unique)->default_value(false), "When grabbing jobs, grab the unique id")
            ("verbose", boost::program_options::bool_switch(&options.verbose)->default_value(true), "Print to stdout information as job is processed.")
            ;


    variables_map vm;
    try {
        store(parse_command_line(args, argv, desc), vm);
        notify(vm);
    }
    catch(exception &e) {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help")) {
        cout << desc << endl;
        return EXIT_SUCCESS;
    }

    //  Save a copy to the worker options  D. Ison 5-2015
 //   cout << "**Host coming in as " << host << endl;
    options.host = host;


    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
        cerr << "signal:" << strerror(errno) << endl;
        return EXIT_FAILURE;
    }

    gearman_worker_st *worker;
    if ((worker = gearman_worker_create(NULL)) == NULL) {
        cerr << "Memory allocation failure on worker creation." << endl;
        return EXIT_FAILURE;
    }

    if (options.unique) {
        gearman_worker_add_options(worker, GEARMAN_WORKER_GRAB_UNIQ);
    }

    if (timeout >= 0) {
        gearman_worker_set_timeout(worker, timeout);
    }

    if (gearman_failed(gearman_worker_add_server(worker, host.c_str(), port))) {
        cerr << gearman_worker_error(worker) << endl;
        return EXIT_FAILURE;
    }

    if (identifier.empty() == false) {
        if (gearman_failed(gearman_worker_set_identifier(worker, identifier.c_str(), identifier.size()))) {
            cerr << gearman_worker_error(worker) << endl;
            return EXIT_FAILURE;
        }
    }


    gearman_function_t worker_fn = gearman_function_create(gear_worker);

  //  if (gearman_failed(gearman_worker_define_function(worker, gearman_literal_param("cuda_sqrt"),
      if (gearman_failed(gearman_worker_define_function(worker,
              gearman_literal_param(STRINGIZE_VALUE_OF(WORKER_NAME)),
            worker_fn, 0, &options))) {
        cerr << gearman_worker_error(worker) << endl;
        return EXIT_FAILURE;
    }

    // Add one if count is not zero
    if (limit != 0) {
        limit++;
    }


    while (--limit) {
        if (gearman_failed(gearman_worker_work(worker))) {
            cerr << gearman_worker_error(worker) << endl;
            break;
        }
        cout << "Gearman worker is initializing.." << endl;

    }

    gearman_worker_free(worker);

    return EXIT_SUCCESS;
}
