/**
*   workerConsoleUI.cpp -
*
*   Provides main() entry point and framework for testing GravityNeutral 
*   workers
*
*/


#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <signal.h>

#include <boost/program_options.hpp>
#include <hglaunch.hpp>


// Why type std:: everywhere?
using namespace std;


// Todo:  Can these be generalized for different kernels?
struct worker_options_t {
    bool chunk;
    bool status;
    bool unique;
    bool dump;
    bool verbose;

    worker_options_t():
            chunk(false),
            status(false),
            unique(false),
            verbose(false) { }
};




int main(int args, char *argv[]) {
    uint64_t limit;
    worker_options_t options;
    int timeout;

    string host;
    string identifier;
    // This gets formatted on command line exactly as it would come from gearman side ie --data="hello world"
    string source_data;
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("help", "Options related to the program.")
            ("json,j", boost::program_options::value<string>(&source_data)->default_value("{}"),"JSON data to send to kernel")
            ("identifier", boost::program_options::value<string>(&identifier),"Assign identifier")
            ("verbose", boost::program_options::bool_switch(&options.verbose)->default_value(true), "Print to stdout information as job is processed.")
            ("dump,d", boost::program_options::bool_switch(&options.dump)->default_value(false), "Dump results JSON object.")

            ;

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(args, argv, desc), vm);
        boost::program_options::notify(vm);
    }
    catch(exception &e) {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help")) {
        cout << desc << endl;
        return EXIT_SUCCESS;
    }

    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
        cerr << "signal:" << strerror(errno) << endl;
        return EXIT_FAILURE;
    }

    string job_handle = "cli_task";
    ostringstream results_stream;
/*    HPCgearLaunch(static_cast<const char *> (&source_data[0]), source_data.size(),
                  static_cast<const char *> (&job_handle[0]), results_stream);*/


    HgTaskParams task_params (source_data.c_str(), source_data.length(), job_handle, std::string(""),
                              &results_stream);

    HPCgearLaunch (task_params);






    if (options.dump) {
        cout << results_stream.str() << endl;
    }

}
