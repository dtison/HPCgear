
/**
* Set up the local worker subclass
*/

#include <hgworker.hpp>

// Worker Interface - Pi num sims
typedef struct {
    long long unsigned num_sims;
    bool is_single;
} WORKER_VALUES;

// Subclass
class PiEstimatorWorker: public HgWorker {
public:
    PiEstimatorWorker(HgTaskParams & task_params);
    void operator()(boost::property_tree::ptree & results_properties);
protected:
    WORKER_VALUES workerValues;
};
