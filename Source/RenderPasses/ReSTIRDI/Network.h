#pragma once

#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector_types.h"
#include "DataStructures.slang"
namespace NRCNetwork{

const static ::uint2 tiling = {6, 6};
const static uint32_t n_train_batch = 4;
const static uint32_t n_batch_size = 1 << 14; // 16384

const static uint32_t max_training_size = 1 << 16; // ~57,600

//const uint input_dim = 16;

const static uint32_t n_input_dims = 16;
const static uint32_t n_output_dims = 3;

class Network
{
public:
    Network(unsigned int X, unsigned int Y);
    __host__ void NRCQuery(NRCData::NRCQueryEntry* queryEntry, ::uint2* queryPixel, uint32_t queryCount, cudaSurfaceObject_t output);
    __host__ void NRCTrain(NRCData::NRCTrainingEntry* trainEntry, uint32_t* trainEntryCount);
    void reset();

private:
    uint32_t seed = 11086u;
    float learning_rate = 1e-4f;
    uint64_t train_times = 0;

    ::uint2 mFrameDims;
};
}
