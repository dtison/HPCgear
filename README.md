# HPCgear  
Library to implement HPC Command line applications as Gearman workers.

Library supports CUDA and Intel MIC.

# What is HPCgear?

HPCgear is a C++ library framework designed to enable conversion of command line applications to web-server capable entities known as Gearman workers.  All you have to do is re-align your application's job parameters to web supported JSON format, and change output from stdout to a simple API that uses message queuing (zeromq library) to communicate with the web server.  From the web server, the UI can be implemented in a browser, for example using Server-Sent Events. (SSE).

# Gearman Workers
If you already know what Gearman is, you'll understand how implementing your HPC app as a Gearman worker opens up a lot of benefits for UI, etc.  If you don't know what is Gearman, click below.

[![N|Solid](http://gearman.org/img/logo.png)](http://gearman.org)
>Gearman provides a generic application framework to farm out work to other machines or processes.   
HPCgear Library provides the means to deploy your HPC application as a Gearman worker.

You install Gearman on a server, and you can start submitting jobs to it using the Gearman API.
Your HPC application will pick up the task and execute it automatically.

### Scenarios:
  - You want a web-based UI for your HPC application.

### Other Use Cases:
  - You want to put a web server in front of your CUDA kernels.
  -  Microservices Architecture or SOA on a web server integrated with your HPC applications.
  - You want to export HPC compute capability through a REST API.
  - You want to put a web server in front of your CUDA kernels.
 



# System Requirements

HPCgear Library has the following dependencies:

| Library | URL |
| ------ | ------ |
| Zeromq | [www.zeromq.org] [PlDb] |
| Gearman | [http://gearman.org/] [PlGh] |
| CUDA SDK | [https://developer.nvidia.com/cuda-downloads] [PlGd] |
| CMake | [https://cmake.org/download/] [PlGd] |

Your OS might have an easier way to install these, e.g. Homebrew on OSX, or apt-get on Ubuntu.
I don't know about MS Windows systems, any contributors can help?

Also Intel Knight's Corner / Knight's Landing software stack needs to be added to this list.


# Installation
Library uses CMake for worker builds.  To install and build an example worker:
```sh
$ git clone...
$ cd workers/cudaPiEstimator
$ mkdir build
$ cd build
$ cmake ../
```
# Version
Library is currently beta, v 0.80.

#  Contributing
Contributions and pull requests welcome!
If you have access to Intel Xeon Phi and want to implement a MIC worker that would be excellent.


# About the Author

David Ison has been developing software since the 1980's.  In addition to C++, he programs web applications in PHP / LAMP Stack, and front end Javascript in React.js.  He has been a Linux enthusiast and web developer since 1998 and interested in HPC computing platforms since 2014.



