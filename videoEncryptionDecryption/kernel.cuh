#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <openssl/sha.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//system runtime parameter macros
#define ORIGINAL_VIDEO_FILE_PATH                                    "../originalVideo/coastguard.mp4"
#define NUMBER_OF_WORKER_THREADS                                    8
#define NUMBER_OF_UINT16T_DATA_EXTRACTED_FROM_EACH_ITERATION_RESULT 3
#define NUMBER_OF_BYTES_EXTRACTED_FROM_EACH_ITERATION_RESULT        6
#define PRE_ITERATIONS                                              200
#define GPU_BLOCK_LENGTH                                            128

//structure used to pass parameters to worker threads
struct workerThreadsParameterStructure{
    int             threadIndex;
    int             frameDataLengthForEachWorkerThread;
    unsigned char * SHA256HashResultArray;
    double        * initialConditionArray;
    int             iterationsForGeneratingShiftDistances;
    uint16_t      * shiftDistanceSequence;
    int             iterationsForGeneratingByteSequence;
    unsigned char * byteSequence;
};

//all worker threads execute this function to concurrently generate the data required for encryption and decryption processes
static void * workerThreadFunction(void * arg);

//generate a double-precision floating-point number within the range of minValue and maxValue
double generateRandomDoubleNumber(double minValue, double maxValue);

//get cpu time
double getCPUSecond(void);

//calculate SHA256 hash
void calculateSHA256Hash(unsigned char * frameData, size_t dataLength, unsigned char * SHA256HashResultArray);

//reconstruct the initial condition using the generated SHA2-256 hash
double reconstructInitialCondition(unsigned char * SHA256HashResultArray, double initialCondition, int16_t lowerBound, int16_t upperBound);

//generate initial conditions for initializing LHCSs of the worker threads
void generateInitialConditionsForWorkerThreads(double * x0, double * y0, double * z0, double * w0, double * initialConditionArray);

//iterate Lorenz map and return the results
void iterateLorenzHyperChaoticMap(double * x, double * y, double * z, double * w);

//iterate the hyper chaotic map, generate iteration results, and store the results
void generateIterationResults(double * x, double * y, double * z, double * w, int iterations, double * iterationResultArray);

//convert iteration results into uint16_t data
void convertIterationResultsToUint16Data(double * iterationResultArray, uint16_t * uint16ResultArray, int numberOfIterationsResults);

//convert iteration results into bytes
void convertResultsToBytes(double * iterationResultArray, unsigned char * byteSequence, int numberOfIterationsResults);

//the main thread call this function to encrypt the original frame using GPU
extern "C"
void encryptionKernelCaller(uchar3 * deviceOriginalFrame, uchar3 * deviceEncryptedFrame, uchar3 * deviceEncryptedBitMatrix, uchar3 * deviceTempBitMatrix,
                            uint16_t * deviceShiftDistanceSequence, unsigned char * deviceByteSequence, int frameWidth, int frameHeight, int frameDataLength);

//the main thread call this function to decrypt the encrypted frame using GPU
extern "C"
void decryptionKernelCaller(uchar3 * deviceEncryptedFrame, uchar3 * deviceDecryptedFrame, uchar3 * deviceDecryptedBitMatrix, uchar3 * deviceTempBitMatrix,
                            uint16_t * deviceShiftDistanceSequence, unsigned char * deviceByteSequence, int frameWidth, int frameHeight, int frameDataLength);

#endif