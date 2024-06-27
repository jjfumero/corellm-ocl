// PoC for core functions from LLama2.c in OpenCL
// This code just shows a quick demo

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

#ifdef __APPLE
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

// =============================
// Configuration
// =============================
const int PLATFORM = 1;
const long elements = 4096;
const int WORK_GROUP_SIZE = 1024;
// =============================

// Variables
// Host Objects
float *hOutput;
float *hX;
float *hWeight;
float *hXout;
float *hW;

// Device Memory Objects
cl_mem dOutput;
cl_mem dX;
cl_mem dWeight;
cl_mem dXout;
cl_mem dW;

// OpenCL env.
cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
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
    cl_kernel kernel = clCreateKernel(program, kernelName, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateKernel" << endl;
        return nullptr;
    }
    return kernel;
}


void hostDataInitialization(long elements) {
    const long data_size = elements * sizeof(float);
    hOutput = static_cast<float *>(malloc(data_size));
    hX = static_cast<float *>(malloc(data_size));
    hWeight = static_cast<float *>(malloc(data_size));
    hXout = static_cast<float *>(malloc(data_size));
    hW = static_cast<float *>(malloc(elements * elements * sizeof(float)));
}

cl_int allocateBuffersOnGPU(long elements) {
    long data_size = elements * sizeof(float);
    cl_int status;
    dOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateBuffer (dOutput)" << endl;
        return -1;
    }
    dX      = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateBuffer (dX)" << endl;
        return -1;
    }
    dWeight = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateBuffer (dWeight)" << endl;
        return -1;
    }
    dXout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateBuffer (dWeight)" << endl;
        return -1;
    }
    // Matrix
    dW = clCreateBuffer(context, CL_MEM_WRITE_ONLY, elements * elements * sizeof(float), nullptr, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateBuffer (dOutput)" << endl;
        return -1;
    }
    return status;
}

cl_event writeBuffer(cl_mem dVar, float* hostVar, int data_size) {
    cl_event writeEvent;
    cl_int status = clEnqueueWriteBuffer(commandQueue, dVar, CL_TRUE, 0, data_size, hostVar, 0, nullptr, &writeEvent);
    if (status != CL_SUCCESS) {
        cout << "Error in clEnqueueWriteBuffer: " << status << endl;
        return nullptr;
    }
    return writeEvent;
}

cl_event runKernel1(cl_kernel kernel, ulong elements) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dOutput);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
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

cl_event runKernel2(cl_kernel kernel, ulong elements, float ss) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dOutput);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dWeight);
    status |= clSetKernelArg(kernel, 3, sizeof(float), &ss);
    if (status != CL_SUCCESS) {
        cout << "Error in clSetKernelArg" << endl;
        return nullptr;
    }
    size_t globalWorkSize[] = {elements, 1, 1};
    cl_event kernelEvent;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernelEvent);
    if (status != CL_SUCCESS) {
        cout << "Error clEnqueueNDRangeKernel: "  << status << endl;
        return nullptr;
    }
    return kernelEvent;
}

cl_event runKernel4(cl_kernel kernel, ulong elements, float max) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dOutput);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
    int shared_size = sizeof(float) * WORK_GROUP_SIZE;
    status |= clSetKernelArg(kernel, 2, shared_size, nullptr);
    status |= clSetKernelArg(kernel, 3, sizeof(float), &max);
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

cl_event runKernel5(cl_kernel kernel, ulong elements, float sums) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dX);
    status |= clSetKernelArg(kernel, 1, sizeof(float), &sums);
    if (status != CL_SUCCESS) {
        cout << "Error in clSetKernelArg" << endl;
        return nullptr;
    }
    size_t globalWorkSize[] = {elements, 1, 1};
    cl_event kernelEvent;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernelEvent);
    if (status != CL_SUCCESS) {
        cout << "Error clEnqueueNDRangeKernel: "  << status << endl;
        return nullptr;
    }
    return kernelEvent;
}

cl_event runKernel6(cl_kernel kernel, ulong elements) {
    cl_int status;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dXout);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dX);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dW);
    status |= clSetKernelArg(kernel, 3, sizeof(long), &elements);
    if (status != CL_SUCCESS) {
        cout << "Error in clSetKernelArg" << endl;
        return nullptr;
    }
    size_t globalWorkSize[] = {elements, 1, 1};
    cl_event kernelEvent;
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, &kernelEvent);
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


void runRmsNorm(long elements, cl_kernel kernel1, cl_kernel kernel2) {
    long dataSize = elements * sizeof(float);
    writeBuffer(dOutput, hOutput, dataSize);
    writeBuffer(dX, hX, dataSize);

    cl_event kernelEvent = runKernel1(kernel1, elements);
    read(dOutput, hOutput, dataSize, kernelEvent);

    int numGroups = elements / WORK_GROUP_SIZE;
    for (int i = 0; i < numGroups; i++) {
        hOutput[0] += hOutput[i];
    }
    float ss = hOutput[0] + 1e-5;
    ss = 1.0 / sqrt(ss);
    cout << "SS: " << ss << endl;

    writeBuffer(dOutput, hOutput, dataSize);
    writeBuffer(dX, hX, dataSize);
    writeBuffer(dWeight, hWeight, dataSize);

    kernelEvent = runKernel2(kernel2, elements, ss);
    read(dOutput, hOutput, dataSize, kernelEvent);
}

void runSoftMax(const long elements, cl_kernel kernel1, cl_kernel kernel2, cl_kernel kernel3) {
    long dataSize = elements * sizeof(float);
    writeBuffer(dOutput, hOutput, dataSize);
    writeBuffer(dX, hX, dataSize);

    cl_event kernelEvent = runKernel1(kernel1, elements);
    read(dOutput, hOutput, dataSize, kernelEvent);

    int numGroups = elements / WORK_GROUP_SIZE;
    for (int i = 0; i < numGroups; i++) {
        if (hOutput[0] < hOutput[i]) {
            hOutput[0] = hOutput[i];
        }
    }
    float max = hOutput[0];

    writeBuffer(dOutput, hOutput, dataSize);
    writeBuffer(dX, hX, dataSize);
    writeBuffer(dWeight, hWeight, dataSize);

    kernelEvent = runKernel4(kernel2, elements, max);
    read(dOutput, hOutput, dataSize, kernelEvent);

    for (int i = 1; i < numGroups; i++) {
        hOutput[0] += hOutput[i];
    }
    float sum = hOutput[0];

    kernelEvent = runKernel5(kernel3, elements, sum);
    read(dX, hX, dataSize, kernelEvent);
}

void runMatMul(const long elements, cl_kernel kernel1) {
    long dataSize = elements * sizeof(float);
    writeBuffer(dXout, hXout, dataSize);
    writeBuffer(dX, hX, dataSize);
    writeBuffer(dW, hW, sizeof(float) * elements * elements);
    cl_event kernelEvent = runKernel6(kernel1, elements);
    read(dXout, hXout, dataSize, kernelEvent);
}

void free(const std::vector<cl_kernel>& kernels, const std::vector<cl_mem>& deviceObjects, const std::vector<void *>& hostObjects) {

    for (auto kernel: kernels)
        clReleaseKernel(kernel);

    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);

    for (auto object: deviceObjects)
        clReleaseMemObject(object);

    clReleaseContext(context);
    clReleaseContext(context);
    free(platforms);
    free(devices);

    for (auto object: hostObjects)
        free(object);
}

float randValue() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

/**
 * OpenCL program to accelerate the core LLM functions
 */
int main(int argc, char** argv) {

    cout << "LLM Llama3 Core Math Library" << endl;
    initOpenCLPlatformAndKernels();

    hostDataInitialization(elements);
    for (int i = 0; i < elements; i++) {
        hOutput[i] = 0;
        hX[i] = randValue();
        hWeight[i] = randValue();
        for (int j = 0; j < elements; j++) {
            hW[i * elements + j] = randValue();
        }
    }

    allocateBuffersOnGPU(elements);

    // rmsNorm kernels
    cl_kernel kernel1 = createKernel("rmsnormReduction");
    cl_kernel kernel2 = createKernel("rmsnormNormalization");

    runRmsNorm(elements, kernel1, kernel2);

    // softMax Kernels
    cl_kernel kernel3 = createKernel("softMaxReduction");
    cl_kernel kernel4 = createKernel("softMaxExpAndSum");
    cl_kernel kernel5 = createKernel("softMaxNormalization");
    runSoftMax(elements, kernel3, kernel4, kernel5);

    // matMul
    cl_kernel kernel6 = createKernel("matMul");
    runMatMul(elements, kernel6);

    // free environment
    vector<cl_kernel> kernels = { kernel1, kernel2, kernel3, kernel4, kernel5, kernel6};
    vector<cl_mem> deviceObjects = { dOutput, dX, dXout, dWeight, dW };
    vector<void *> hostObjects = { hOutput, hX, hWeight, hXout, hW };
    free(kernels, deviceObjects, hostObjects);

    return 0;
}
