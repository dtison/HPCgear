//
// Created by David Ison on 5/25/15.
//

#ifndef GRAVITYNEUTRAL_GNLAUNCH_HPP
#define GRAVITYNEUTRAL_GNLAUNCH_HPP

class GNTaskParams  {
public:
    GNTaskParams (const char * source_data, const unsigned int source_len, const std::string job_handle,
            const std::string host, std::ostringstream * results_stream) :
            source_data(source_data), source_len(source_len), job_handle(job_handle),
            host(host) , results_stream(results_stream)  {};
private:
    //  JSON object with all needed parameters tos this job.  Port property added by web server
    const char * source_data;
    //  Length of source_data
    const unsigned int source_len;
    //  Gearman job_handle
    const std::string job_handle;
    //  Host parameter specified on CLI (obtained via Gearman)
    const std::string host;

public:
    //  Output results stream, used only for cli mode and
    //  jobs sent by web server that wait for results.  (Not used in server push mode)
    //  Moved to public access because - using getter failed for write_json() call
    std::ostringstream * results_stream;

    char const *get_source_data() const {
        return source_data;
    }
    std::string const get_host() const {
        return host;
    }
    std::string const get_job_handle() const {
        return job_handle;
    }
};



//  Worker interface functions
extern "C" bool gravityLaunch(GNTaskParams & task_params);



#endif //GRAVITYNEUTRAL_GNLAUNCH_HPP
