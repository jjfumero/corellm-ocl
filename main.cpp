#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#ifdef __APPLE
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

const int PLATFORM = 1;
const int WORK_GROUP_SIZE = 1024;

// Variables
size_t data_size;
float *hA;
float *hB;
float *hC;

cl_mem dA;
cl_mem dB;
cl_mem dC;

cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernelPtr;
cl_program program;

char *readsource(const char *sourceFilename) {

    FILE *fp;
    int err;
    int size;
    char *source;

    fp = fopen(sourceFilename, "rb");

    if(fp == NULL) {
        printf("Could not open kernelPtr file: %s\n", sourceFilename);
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_END);

    if(err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);

    }
    size = ftell(fp);

    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_SET);
    if(err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);

    }

    source = (char*)malloc(size+1);

    if(source == NULL) {
        printf("Error allocating %d bytes for the program source\n", size+1);
        exit(-1);
    }

    err = fread(source, 1, size, fp);
    if(err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }

    source[size] = '\0';
    return source;
}

int initOpenCLPlatformAndKernels() {
    cl_int status;
    cl_uint numPlatforms = 0;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0) {
        cout << "No OpenCL platform detected" << endl;
        return -1;
    }

    platforms = (cl_platform_id*) malloc(numPlatforms*sizeof(cl_platform_id));
    if (platforms == nullptr) {
        cout << "Malloc Platforms failed" << endl;
        return -1;
    }

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        cout << "clGetPlatformIDs failed" << endl;
        return -1;
    }

    cout << numPlatforms <<  " has been detected" << endl;
    string platformName = "";
    for (int i = 0; i < numPlatforms; i++) {
        char buf[10000];
        cout << "Platform: " << i << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        if (i == PLATFORM) {
            platformName += buf;
        }
        cout << "\tVendor: " << buf << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    }

    cl_uint numDevices = 0;
    cl_platform_id platform = platforms[PLATFORM];
    std::cout << "Using platform: " << PLATFORM << " --> " << platformName << std::endl;

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

    if (status != CL_SUCCESS) {
        cout << "[WARNING] Using CPU, no GPU available" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    } else {
        devices = (cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }

    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if (context == NULL) {
        cout << "Context is not NULL" << endl;
    }

    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS || commandQueue == NULL) {
        cout << "Error in create command" << endl;
        return -1;
    }

    const char *sourceFile = "kernels.cl";
    char *source = readsource(sourceFile);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &status);

    cl_int buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    return 0;
}

cl_kernel createKernel(const char* kernelName) {
    cl_int status;
    kernelPtr = clCreateKernel(program, kernelName, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateKernel" << endl;
        return nullptr;
    }
    return kernelPtr;
}


void hostDataInitialization(long data_size) {
    hA = static_cast<float *>(malloc(data_size));
    hB = static_cast<float *>(malloc(data_size));
    hC = static_cast<float *>(malloc(data_size));
}

cl_int allocateBuffersOnGPU(int data_size) {
    cl_int status;
    dA = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);
    dB = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr, &status);
    dC = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr, &status);
    return status;
}

cl_event writeBuffer(cl_mem dVar, float* hostVar, int data_size) {
    cl_event writeEvent;
    clEnqueueWriteBuffer(commandQueue, dVar, CL_TRUE, 0, data_size, hostVar, 0, nullptr, &writeEvent);
    return writeEvent;
}

cl_event runKernel1(cl_kernel kernel, ulong elements) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    int shared_size = sizeof(float) * WORK_GROUP_SIZE;
    status |= clSetKernelArg(kernel, 2, shared_size, NULL);
    if (status != CL_SUCCESS) {
        cout << "Error in clSetKernelArg" << endl;
        return nullptr;
    }

    size_t globalWorkSize[] = {elements, 1, 1};
    size_t localWorkSize[]  = { WORK_GROUP_SIZE, 1, 1 };
    cl_event kernelEvent;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &kernelEvent);
    if (status != CL_SUCCESS) {
        cout << "Error clEnqueueNDRangeKernel: "  << status << endl;
        return nullptr;
    }

    return kernelEvent;
}

cl_event read(cl_mem devVar, float* hostVar, int data_size, cl_event kernelEvent) {
    // Read result back from device to host
    cl_event waitEventsRead[] = {kernelEvent};
    cl_event readEvent;
    clEnqueueReadBuffer(commandQueue, devVar, CL_TRUE, 0, data_size, hostVar, 1, waitEventsRead , &readEvent);
    return readEvent;
}

/**
 * OpenCL program to accelerate the core LLM functions
 */
int main(int argc, char** argv) {

    cout << "LLM Llama3 Core Math Library" << endl;
    initOpenCLPlatformAndKernels();

    const long elements = 4096;

    cl_kernel kernel1 = createKernel("rmsnormReduction");
    cl_kernel kernel2 = createKernel("rmsnormNormalization");
    cl_kernel kernel3 = createKernel("softMaxReduction");
    cl_kernel kernel4 = createKernel("softMaxExpAndSum");
    cl_kernel kernel5 = createKernel("softMaxNormalization");
    cl_kernel kernel6 = createKernel("matMul");
    data_size = sizeof(float) * elements;
    hostDataInitialization(data_size);
    for (int i = 0; i < elements; i++) {
        hA[i] = 0.1;
        hB[i] = 0.2;
    }
    allocateBuffersOnGPU(data_size);

    writeBuffer(dA, hA, data_size);
    writeBuffer(dB, hB, data_size);

    cl_event kernelEvent = runKernel1(kernel1, elements);
    read(dA, hA, data_size, kernelEvent);

    for (int i = 0; i < elements; i++) {
        cout << hA[i] << " ";
    }

    return 0;
}