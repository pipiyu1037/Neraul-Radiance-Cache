#include "Network.h"
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <json/json.hpp>
#include <iostream>
#include <cstdio>

using namespace tcnn;
using precision_t = tcnn::network_precision_t;

namespace NRCNetwork
{
json config = {
    {"loss", {{"otype", "RelativeL2"}}},
    {"optimizer",
     {
         {"otype", "Ema"},
         {"decay", 0.99},
         {"nested",
               {
                   {"otype", "Adam"},
                   {"learning_rate", 4.5e-3},
                   {"beta1", 0.9},
                   {"beta2", 0.99},
                   {"epsilon", 1e-8},
                   {"l2_reg", 1e-8}
               }
         }
         
     }
    },
    {"encoding",
     {
         {"otype", "Composite"},
         {"nested",
          {
              {
                  {"n_dims_to_encode", 3},
                  {"otype", "TriangleWave"},
                  {"n_frequencies", 12},
              },
              {
                  {"n_dims_to_encode", 5},
                  {"otype", "OneBlob"},
                  {"n_bins", 4},
              },
              {
                  {"otype", "Identity"},
              },
          }},
     }},
    {"network",
     {
         {"otype", "FullyFusedMLP"},
         {"activation", "ReLU"},
         {"output_activation", "None"},
         {"n_neurons", 64},
         {"n_hidden_layers", 5},
     }},
};

json encoding_opts;
json loss_opts;
json optimizer_opts;
json network_opts;

std::shared_ptr<Loss<precision_t>> loss;
std::shared_ptr<Optimizer<precision_t>> optimizer;

cudaStream_t inference_stream;
cudaStream_t train_stream;

GPUMatrix<float> trainTarget;
GPUMatrix<float> trainBatch;

GPUMatrix<float> prediction;
GPUMatrix<float> inferenceBatch;

GPUMemory<float> randomIndex;
curandGenerator_t rng;

std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;
std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

__device__ float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ float3 operator*(float3 a, float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator/(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ float3 safe_div(float3 a, float3 b)
{
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

__device__ void safe_num(float3& num)
{
    num.x = isinf(num.x) || isnan(num.x) ? 0 : num.x;
    num.y = isinf(num.y) || isnan(num.y) ? 0 : num.y;
    num.z = isinf(num.z) || isnan(num.z) ? 0 : num.z;
}

template<typename T>
__device__ void copyEntry(T* data, const NRCData::NRCQueryEntry* query)
{
    const size_t size = sizeof(NRCData::NRCQueryEntry);
    memcpy(data, query, size);
}

void printTrainEntry(const NRCData::NRCTrainingEntry sample)
{
    printf("[%u]|radiance : {%3.2f %3.2f %3.2f}|  ", sample.idx, sample.radiance.x, sample.radiance.y, sample.radiance.z);
    printf("|thp : {%3.2f %3.2f %3.2f}|  --  ", sample.thp.x, sample.thp.y, sample.thp.z);
    printf("|hit_pos : {%3.2f %3.2f %3.2f}|  ", sample.query.pos.x, sample.query.pos.y, sample.query.pos.z);
    printf("|dir : {%3.2f %3.2f}|  ", sample.query.dir.x, sample.query.dir.y);
    printf("|nor : {%3.2f %3.2f}|  ", sample.query.normal.x, sample.query.normal.y);
}

uint32_t showMsg_counter(uint32_t* dataOnDevice)
{
    uint32_t* dataOnHost = new uint32_t[1];
    cudaMemcpy(dataOnHost, dataOnDevice, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("%u\n", dataOnHost[0]);
    uint32_t res = dataOnHost[0];
    delete[] dataOnHost;
    return res;
}

void showMsgColor(const float3* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    float3* dataOnHost = new float3[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printf("{%4.2f %4.2f %4.2f}\n", dataOnHost[i].x, dataOnHost[i].y, dataOnHost[i].z);
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showMsgSample(const NRCData::NRCTrainingEntry* dataOnDevice, int size, int maxsize = 8)
{
    if (size > maxsize)
        size = maxsize;
    NRCData::NRCTrainingEntry* dataOnHost = new NRCData::NRCTrainingEntry[size];
    cudaMemcpy(dataOnHost, dataOnDevice, size * sizeof(NRCData::NRCTrainingEntry), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        printTrainEntry(dataOnHost[i]);
        std::cout << "\n";
    }
    std::cout << std::endl;
    delete[] dataOnHost;
}

void showPixel(const uint2* pixel, int size, int maxSize = 8) {
    if (size > maxSize)
    {
        size = maxSize;
    }
    uint2* pixelOnHost = new uint2[size];
    cudaMemcpy(pixelOnHost, pixel, size * sizeof(uint2), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        printf("x: %d, y: %d\n", pixelOnHost[i].x, pixelOnHost[i].y);
    }
    printf("\n");
    delete[] pixelOnHost;
}
Network::Network(unsigned int X, unsigned int Y)
{
    mFrameDims.x = X;
    mFrameDims.y = Y;
    encoding_opts = config.value("encoding", json::object());
    loss_opts = config.value("loss", json::object());
    optimizer_opts = config.value("optimizer", json::object());
    network_opts = config.value("network", json::object());

    std::shared_ptr<Loss<precision_t>> lossTemp{create_loss<precision_t>(loss_opts)};
    loss = lossTemp;
    std::shared_ptr<Optimizer<precision_t>> optTemp{create_optimizer<precision_t>(optimizer_opts)};
    optimizer = optTemp;

    network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        NRCNetwork::n_input_dims, NRCNetwork::n_output_dims, encoding_opts, network_opts
    );
    trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

    trainTarget = GPUMatrix<float>(NRCNetwork::n_output_dims, n_batch_size);
    trainBatch = GPUMatrix<float>(NRCNetwork::n_input_dims, n_batch_size);

    uint32_t pixelCount = mFrameDims.x * mFrameDims.y;
    prediction = GPUMatrix<float>(NRCNetwork::n_output_dims, pixelCount);
    inferenceBatch = GPUMatrix<float>(NRCNetwork::n_input_dims, pixelCount);
    randomIndex = GPUMemory<float>(n_train_batch * n_batch_size);

    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&train_stream));

    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 10086ULL);
    curandSetStream(rng, train_stream);
    curandGenerateUniform(rng, randomIndex.data(), n_train_batch * n_batch_size);
}

template<uint32_t inputDim, typename T = float>
__global__ void generateTrainBatchAndTarget(
    uint32_t n_elements,
    uint32_t offset,
    NRCData::NRCTrainingEntry* trainEnrty,
    T* trainBatch,
    T* trainTarget,
    uint32_t* trainEnrtyCount,
    float* randomIndex
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;

    uint32_t BatchIndex = i * inputDim;
    uint32_t trainEntryIndex = offset + i;
    if (randomIndex != nullptr)
    {   
        trainEntryIndex = (1 - randomIndex[trainEntryIndex]) * (*trainEnrtyCount);
    }

    if (trainEntryIndex < *trainEnrtyCount)
    {
        float3 radiance = trainEnrty[trainEntryIndex].radiance;
        uint32_t targetIndex = i * 3;
        //memcpy(&trainBatch[BatchIndex], &(trainEnrty[trainEntryIndex].query), sizeof(NRCData::NRCQueryEntry));
        //memcpy(&trainTarget[targetIndex], &trainEnrty[trainEntryIndex].radiance, sizeof(float3));
        copyEntry(&trainBatch[BatchIndex], &trainEnrty[trainEntryIndex].query);
        safe_num(radiance);
        *(float3*)&trainTarget[targetIndex] = radiance;

    }
    return;
}

void Network::NRCTrain(NRCData::NRCTrainingEntry* trainEntry, uint32_t* trainEntryCount)
{
    optimizer.get()->set_learning_rate(learning_rate);
    //showMsgSample(trainEntry + 2001, 6);
    curandGenerateUniform(rng, randomIndex.data(), n_train_batch * n_batch_size);
    float lo = 0.f;

    for (int i = 0; i < n_train_batch; i++)
    {

        tcnn::linear_kernel(
            generateTrainBatchAndTarget<n_input_dims, float>,
            0,
            train_stream,
            n_batch_size,
            i * n_batch_size,
            trainEntry,
            trainBatch.data(),
            trainTarget.data(),
            trainEntryCount,
            randomIndex.data()
        );
        auto ctx = trainer->training_step(train_stream, trainBatch, trainTarget);
        lo += trainer->loss(train_stream, *ctx);
    }
    //printf("loss: %4.2f \n", lo);

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(train_stream);
    if (cudaStatus != cudaSuccess)
    {
        printf("CUDA training error: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    return;
}

template<uint32_t inputDim, typename T = float>
__global__ void generateQueryData(uint32_t queryDataCount, uint32_t offset, NRCData::NRCQueryEntry* queryEntry, T* queryData)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= queryDataCount)
        return;

    uint32_t queryDataIndex = i * inputDim;
    uint32_t queryEntryIndex = offset + i;
    //memcpy(&queryData[queryDataIndex], &queryEntry[queryEntryIndex], sizeof(NRCData::NRCQueryEntry));
    copyEntry(&queryData[queryDataIndex], &queryEntry[queryEntryIndex]);
    return;
}

template<typename T = float>
__global__ void writePredictionToTex(uint32_t queryCount, T* predictionResult, uint2* pixel, cudaSurfaceObject_t output)
{
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i > queryCount)
        return;

    uint32_t pX = pixel[i].x, pY = pixel[i].y;
    uint32_t resultIndex = i * 3;
    float4 radiance = {predictionResult[resultIndex], predictionResult[resultIndex + 1], predictionResult[resultIndex + 2], 1.0f};
    surf2Dwrite(radiance, output, sizeof(float4) * pX, pY);

}

void Network::NRCQuery(NRCData::NRCQueryEntry* queryEntry, uint2* queryPixel, uint32_t queryCount, cudaSurfaceObject_t output)
{
    if (!queryCount)
        return;

    uint32_t queryCountPadded = next_multiple(queryCount, BATCH_SIZE_GRANULARITY);
    inferenceBatch.set_size_unsafe(n_input_dims, queryCountPadded);
    prediction.set_size_unsafe(n_output_dims, queryCountPadded);

    tcnn::linear_kernel(generateQueryData<n_input_dims, float>, 0, inference_stream, queryCount, 0, queryEntry, inferenceBatch.data());

    network.get()->inference(inference_stream, inferenceBatch, prediction);

    tcnn::linear_kernel(writePredictionToTex<float>, 0, inference_stream, queryCount, prediction.data(), queryPixel, output);
    
    cudaError_t cudaStatus;
    cudaStatus = cudaStreamSynchronize(inference_stream);
    if (cudaStatus != cudaSuccess)
    {
        printf("CUDA inference error: %s\n", cudaGetErrorString(cudaStatus));
        exit(-1);
    }
}   

void Network::reset() {
    CUDA_CHECK_THROW(cudaStreamSynchronize(train_stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(inference_stream));

    trainer.get()->initialize_params();
}

}
