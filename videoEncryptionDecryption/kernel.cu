#include "kernel.cuh"

//convert the pixel into eight bits
__global__ void bitExtractation(uchar3 * deviceFrame, uchar3 * deviceTempBitMatrix, int frameWidth){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    //blue channel
    if(threadIndexZ == 0){
        unsigned char pixelValue = deviceFrame[threadIndexY * frameWidth + threadIndexX].x;

        for(int i = 0; i < 8; i ++){
            unsigned char temp                                                        = pixelValue;
            deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x = temp & 0x01;
            pixelValue                                                                = pixelValue >> 1;
        }
    }

    //green channel
    else if(threadIndexZ == 1){
        unsigned char pixelValue = deviceFrame[threadIndexY * frameWidth + threadIndexX].y;

        for(int i = 0; i < 8; i ++){
            unsigned char temp                                                        = pixelValue;
            deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y = temp & 0x01;
            pixelValue                                                                = pixelValue >> 1;
        }
    }

    //red channel
    else{
        unsigned char pixelValue = deviceFrame[threadIndexY * frameWidth + threadIndexX].z;

        for(int i = 0; i < 8; i ++){
            unsigned char temp                                                        = pixelValue;
            deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z = temp & 0x01;
            pixelValue                                                                = pixelValue >> 1;
        }
    }
}

//reconstruct pixel using bits
__global__ void pixelReconstruction(uchar3 * deviceTempBitMatrix, uchar3 * deviceFrame, int frameWidth){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    unsigned char pixelValue = 0;

    //blue channel
    if(threadIndexZ == 0){
        for(int i = 7; i >=0; i --){
            pixelValue = pixelValue | deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x;
            if(i != 0)
                pixelValue = pixelValue << 1;
        }

        deviceFrame[threadIndexY * frameWidth + threadIndexX].x = pixelValue;
    }

    //green channel
    else if(threadIndexZ == 1){
        for(int i = 7; i >=0; i --){
            pixelValue = pixelValue | deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y;
            if(i != 0)
                pixelValue = pixelValue << 1;
        }

        deviceFrame[threadIndexY * frameWidth + threadIndexX].y = pixelValue;
    }

    //red channel
    else{
        for(int i = 7; i >=0; i --){
            pixelValue = pixelValue | deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z;
            if(i != 0)
                pixelValue = pixelValue << 1;
        }

        deviceFrame[threadIndexY * frameWidth + threadIndexX].z = pixelValue;
    }
}

//perform confusion operations along horizontal direction
__global__ void confusionAlongHorizontalDirection(uchar3 * deviceTempBitMatrix, uchar3 * deviceEncryptedBitMatrix, 
                                                  uint16_t * deviceShiftDistanceSequence, int frameWidth){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    //blue channel
    if(threadIndexZ == 0){
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3)] % frameWidth;
        int newThreadIndexX    = (threadIndexX + shiftDistance) % frameWidth;

        for(int i = 0; i < 8; i++)
            deviceEncryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].x = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x;
    }

    //green channel
    else if(threadIndexZ == 1){
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3) + 1] % frameWidth;
        int newThreadIndexX    = (threadIndexX + shiftDistance) % frameWidth;

        for(int i = 0; i < 8; i++)
            deviceEncryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].y = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y;
    }

    //red channel
    else{
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3) + 2] % frameWidth;
        int newThreadIndexX    = (threadIndexX + shiftDistance) % frameWidth;

        for(int i = 0; i < 8; i++)
            deviceEncryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].z = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z;
    }
}

//perform inverse confusion operations along horizontal direction
__global__ void inverseConfusionAlongHorizontalDirection(uchar3 * deviceTempBitMatrix, uchar3 * deviceDecryptedBitMatrix,
                                                        uint16_t * deviceShiftDistanceSequence, int frameWidth){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    //blue channel
    if(threadIndexZ == 0){
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3)] % frameWidth;
        int newThreadIndexX    = (threadIndexX - shiftDistance + frameWidth) % frameWidth;

        for(int i = 0; i < 8; i ++)
            deviceDecryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].x = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x;
    }

    //green channel
    else if(threadIndexZ == 1){
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3) + 1] % frameWidth;
        int newThreadIndexX    = (threadIndexX - shiftDistance + frameWidth) % frameWidth;

        for(int i = 0; i < 8; i ++)
            deviceDecryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].y = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y;
    }

    //red channel
    else{
        uint16_t shiftDistance = deviceShiftDistanceSequence[(threadIndexY * 3) + 2] % frameWidth;
        int newThreadIndexX    = (threadIndexX - shiftDistance + frameWidth) % frameWidth;

        for(int i = 0; i < 8; i ++)
            deviceDecryptedBitMatrix[(threadIndexY * frameWidth + newThreadIndexX) * 8 + i].z = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z;
    }
}


//perform confusion operations along vertical direction
__global__ void confusionAlongVerticalDirection(uchar3 * deviceTempBitMatrix, uchar3 * deviceEncryptedBitMatrix,
                                                uint16_t * deviceShiftDistanceSequence, int frameWidth, int frameHeight){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    //blue channel
    if(threadIndexZ == 0){
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3 + (threadIndexX * 3)] % frameHeight;
        int newThreadIndexY    = (threadIndexY + shiftDistance) % frameHeight;

        for(int i = 0; i < 8; i ++)
            deviceEncryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].x = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x;
    }

    //green channel
    else if(threadIndexZ == 1){
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3 + (threadIndexX * 3) + 1] % frameHeight;
        int newThreadIndexY    = (threadIndexY + shiftDistance) % frameHeight;

        for(int i = 0; i < 8; i ++)
            deviceEncryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].y = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y;
    }

    //red channel
    else{
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3 + (threadIndexX * 3) + 2] % frameHeight;
        int newThreadIndexY    = (threadIndexY + shiftDistance) % frameHeight;

        for(int i = 0; i < 8; i ++)
            deviceEncryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].z = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z;
    }
}

//perform inverse confusion operations along vertical direction
__global__ void inverseConfusionAlongVerticalDirection(uchar3 * deviceTempBitMatrix, uchar3 * deviceDecryptedBitMatrix,
                                                       uint16_t * deviceShiftDistanceSequence, int frameWidth, int frameHeight){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    //blue channel
    if(threadIndexZ == 0){
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3 + (threadIndexX * 3)] % frameHeight;
        int newThreadIndexY    = (threadIndexY - shiftDistance + frameHeight) % frameHeight;

        for(int i = 0; i < 8; i++)
            deviceDecryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].x = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x;
    }

    //green channel
    else if(threadIndexZ == 1){
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3 + (threadIndexX * 3) + 1] % frameHeight;
        int newThreadIndexY    = (threadIndexY - shiftDistance + frameHeight) % frameHeight;

        for(int i = 0; i < 8; i++)
            deviceDecryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].y = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y;
    }

    //red channel
    else{
        uint16_t shiftDistance = deviceShiftDistanceSequence[frameHeight * 3+ (threadIndexX * 3) + 2] % frameHeight;
        int newThreadIndexY    = (threadIndexY - shiftDistance + frameHeight) % frameHeight;

        for(int i = 0; i < 8; i++)
            deviceDecryptedBitMatrix[(newThreadIndexY * frameWidth + threadIndexX) * 8 + i].z = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z;
    }
}

//perform XOR operation to encrypt or decrypt the pixel
__global__ void XOROperations(uchar3 * deviceTempBitMatrix, uchar3 * deviceResultBitMatrix, unsigned char * deviceByteSequence, int frameWidth){
    //calculate the thread index
    int threadIndexX = threadIdx.y + blockIdx.y * blockDim.y;
    int threadIndexY = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndexZ = threadIdx.z;

    if(threadIndexZ == 0){
        unsigned char byteValue = deviceByteSequence[(threadIndexY * frameWidth) * 3 + threadIndexX * 3];

        for(int i = 0; i < 8; i ++){
            unsigned char temp = 0;
            temp               = byteValue & 0x01;
            deviceResultBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].x ^ temp;
            byteValue          = byteValue >> 1;
        }
    }

    else if(threadIndexZ == 1){
        unsigned char byteValue = deviceByteSequence[(threadIndexY * frameWidth) * 3 + threadIndexX * 3 + 1];

        for(int i = 0; i < 8; i ++){
            unsigned char temp = 0;
            temp               = byteValue & 0x01;
            deviceResultBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].y ^ temp;
            byteValue          = byteValue >> 1;
        }
    }

    else{
        unsigned char byteValue = deviceByteSequence[(threadIndexY * frameWidth) * 3 + threadIndexX * 3 + 2];

        for(int i = 0; i < 8; i ++){
            unsigned char temp = 0;
            temp               = byteValue & 0x01;
            deviceResultBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z = deviceTempBitMatrix[(threadIndexY * frameWidth + threadIndexX) * 8 + i].z ^ temp;
            byteValue          = byteValue >> 1;
        }
    }
}

//the main thread call this function to encrypt the original frame using GPU
extern "C"
void encryptionKernelCaller(uchar3 * deviceOriginalFrame, uchar3 * deviceEncryptedFrame, uchar3 * deviceEncryptedBitMatrix, uchar3 * deviceTempBitMatrix,
                            uint16_t * deviceShiftDistanceSequence, unsigned char * deviceByteSequence, int frameWidth, int frameHeight, int frameDataLength){

    //Each block consists of (1, GPU_BLOCK_LENGTH, 3) GPU threads, with a total of (frameHeight, frameWidth / GPU_BLOCK_LENGTH) blocks.
    dim3 block(1, GPU_BLOCK_LENGTH, 3);
    dim3 grid(frameHeight, frameWidth / GPU_BLOCK_LENGTH);

    //convert the original frame into bit matrix
    bitExtractation<<<grid, block>>>(deviceOriginalFrame, deviceTempBitMatrix, frameWidth);
    cudaDeviceSynchronize();

    //perform confusion operations along horizontal direction
    confusionAlongHorizontalDirection<<<grid, block>>>(deviceTempBitMatrix, deviceEncryptedBitMatrix, deviceShiftDistanceSequence, frameWidth);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceEncryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //perform confusion operations along vertical direction
    confusionAlongVerticalDirection<<<grid, block>>>(deviceTempBitMatrix, deviceEncryptedBitMatrix, deviceShiftDistanceSequence, frameWidth, frameHeight);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceEncryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //perform XOR operations to encrypt the shuffled bit matrix
    XOROperations<<<grid, block>>>(deviceTempBitMatrix, deviceEncryptedBitMatrix, deviceByteSequence, frameWidth);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceEncryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //convert the encrypted frame into pixel level frame
    pixelReconstruction<<<grid, block>>>(deviceTempBitMatrix, deviceEncryptedFrame, frameWidth);
    cudaDeviceSynchronize();
}

//the main thread call this function to decrypt the encrypted frame using GPU
extern "C"
void decryptionKernelCaller(uchar3 * deviceEncryptedFrame, uchar3 * deviceDecryptedFrame, uchar3 * deviceDecryptedBitMatrix, uchar3 * deviceTempBitMatrix,
                            uint16_t * deviceShiftDistanceSequence, unsigned char * deviceByteSequence, int frameWidth, int frameHeight, int frameDataLength){
    
    //Each block consists of (1, GPU_BLOCK_LENGTH, 3) GPU threads, with a total of (frameHeight, frameWidth / GPU_BLOCK_LENGTH) blocks.
    dim3 block(1, GPU_BLOCK_LENGTH, 3);
    dim3 grid(frameHeight, frameWidth / GPU_BLOCK_LENGTH);

    //convert the encrypted frame into bit matrix
    bitExtractation<<<grid, block>>>(deviceEncryptedFrame, deviceTempBitMatrix, frameWidth);
    cudaDeviceSynchronize();

    //perform XOR operations to decrypt the encrypted bit matrix
    XOROperations<<<grid, block>>>(deviceTempBitMatrix, deviceDecryptedBitMatrix, deviceByteSequence, frameWidth);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceDecryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //perform inverse confusion operations along vertical direction
    inverseConfusionAlongVerticalDirection<<<grid, block>>>(deviceTempBitMatrix, deviceDecryptedBitMatrix, deviceShiftDistanceSequence, frameWidth, frameHeight);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceDecryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //perform inverse confusion operations along horizontal direction
    inverseConfusionAlongHorizontalDirection<<<grid, block>>>(deviceTempBitMatrix, deviceDecryptedBitMatrix, deviceShiftDistanceSequence, frameWidth);
    cudaDeviceSynchronize();
    cudaMemcpy(deviceTempBitMatrix, deviceDecryptedBitMatrix, frameDataLength * 8, cudaMemcpyDeviceToDevice);

    //convert the decrypted frame into pixel level frame
    pixelReconstruction<<<grid, block>>>(deviceTempBitMatrix, deviceDecryptedFrame, frameWidth);
    cudaDeviceSynchronize(); 
}