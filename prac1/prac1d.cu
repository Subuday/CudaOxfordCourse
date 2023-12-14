#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceId = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  int maxBlocksPerSM;
  int maxThreadsPerBlock;
  int maxThreadsPerSM;

  cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor,
                         deviceId);

  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock,
                         deviceId);

  maxThreadsPerSM = maxBlocksPerSM * maxThreadsPerBlock;

  std::cout << "Max Blocks Per SM: " << maxBlocksPerSM << std::endl;
  std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads Per SM (nsize): " << maxThreadsPerSM << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;

  return 0;
}