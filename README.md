## Real-Time Bit-Level Encryption of Full High Definition Video without Diffusion

#### Dong Jiang, Hui-ran Luo, Zi-jian Cui, Xi-jue Zhao, Lin-sheng Huang, and Liang-liang Lu

#### Develoment enviroment

* CPU             : Intel Xeon Gold 6226R @ 2.9GHz
* GPU             : NVIDIA Geforce RTX 3090
* Memory          : 32GB
* Operating system: Ubuntu 22.04
* OpenCV          : 4.5.4
* CUDA            : 12.4

#### File description

* originalVideo            : This directory contains the original video file named "coastguard.mp4".
* videoEncryptionDecryption: This folder stores the source code for the encryption and decryption algorithms of the proposed protocol.
* NPCRAndUACITest          : This directory contains the source code for performing NPCR and UACI tests on the original video.
* demo1.webm               : This demonstration video illustrates the real-time encryption and decryption of a 1920x1080 FHD video.
* demo2.webm               : This demonstration video presents the NPCR and UACI tests conducted on the original video.

### Code functionality description

The code contained in the videoEncryptionDecryption folder executes frame-by-frame encryption and decryption of the coastGuard.mp4 video file located in the originalVideo folder. The original video is configured with a resolution of 512x512 pixels and a frame rate of 24 FPS to accommodate potential variations in hardware platforms.
The system encrypts the original frame, decrypts the encrypted frame, displays both encrypt and decrypt frames, and reports the encryption time of the current frame every ten frames, along with the total number of frames experiencing delays, the corresponding delay rate, and the average encryption time for all encrypted frames.
A frame is classified as a delayed frame if its encryption time exceeds 1000(ms)/FPS, and the delay rate is calculated by dividing the number of delayed frames by the total number of frames.

The code contained in the NPCRAndUACITest folder conducts frame-by-frame NPCR and UACI tests on the coastGuard.mp4 video file located in the originalVideo folder. It encrypts the original frame to generate an encrypted frame, then randomly selects a channel and a pixel from the original frame, modifies the pixel value by adding a randomly selected increment, and encrypts the modified original frame using the same key to produce another encrypted frame.
It displays the original frame, two encrypted frames, and a corresponding differential map, where a pixel is set to white if the pixel values in any channel at the corresponding positions of the two encrypted frames are equal.
It also calculates and displays the NPCR and UACI between the two generated encrypted frames, along with the average NPCR and UACI.

#### Description of algorithm runtime parameter settings

If the algorithm's runtime environment is correctly configured, executing ./make.sh within the source code directory will automatically compile and run the program. However, to test videos of varying resolutions, it is essential to ensure that the program parameters adhere to the following conditions; otherwise, the program will fail during frame format validation.

* The frame must have 3 channels and a bit depth of 8.
* The data length of the frame (width $\times$ height $\times$ 3) must be evenly divisible by the number of worker threads.
* The frame width must be evenly divisible by GPU_BLOCK_LENGTH.

The macros for algorithm runtime parameters, including the number of worker threads, are defined in the file "kernel.cuh" as follows. Modifications to these macros can be made to adapt the proposed algorithm for videos of varying resolutions.

```
#define ORIGINAL_VIDEO_FILE_PATH "../originalVideo/coastguard.mp4"
#define NUMBER_OF_WORKER_THREADS 8
#define GPU_BLOCK_LENGTH         128
```

