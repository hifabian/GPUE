#include "easyloggingcpp/easylogging++.h"

// Initialize logging system
INITIALIZE_EASYLOGGINGPP

int main(int argc, const char** argv) {
    // Load configuration from file
    el::Configurations conf("src/logConfig.config");
    // Reconfigure single logger
    el::Loggers::reconfigureLogger("default", conf);
    // Check command line for log-relevant flags
    START_EASYLOGGINGPP(argc, argv);
    // Log program title
    LOG(INFO) << argv[0] << " has started successfully\n";
}
