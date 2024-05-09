#pragma once

#include "Falcor.h"
#include "Network.h"
#include "Utils/CudaUtils.h"
#include "vector_types.h"
#include "DataStructures.slang"

enum viewMode
{
    DEFAULT = 0,
    NO_NRC = 1,
    ONLY_QUERY = 2,
    ONLY_FACTOR = 3,
    ONLY_NRC = 4,
};

const Falcor::Gui::DropdownList kNRCViewModeList = {
    {(Falcor::uint)viewMode::DEFAULT, "with NRC"},
    {(Falcor::uint)viewMode::NO_NRC, "NO NRC"},
    {(Falcor::uint)viewMode::ONLY_QUERY, "ONLY QUERY"},
    {(Falcor::uint)viewMode::ONLY_FACTOR, "ONLY FACTOR"},
    {(Falcor::uint)viewMode::ONLY_NRC, "QUERY * FACTOR"},
};

class NRC
{
public:
    NRC(Falcor::ref<Falcor::Device> pDevice, unsigned int FrameDimX, unsigned int FrameDimY);
    void train();
    void query();
    void reset();
    void bindShaderData(const Falcor::ShaderVar& var);
    bool renderUI(Falcor::Gui::Widgets& widget);

    Falcor::ref<Falcor::Device> mpDevice;
    Falcor::ref<Falcor::Buffer> mpTrainingData;
    Falcor::ref<Falcor::Buffer> mpQueryData;
    Falcor::ref<Falcor::Buffer> mpQueryPixel;

    Falcor::ref<Falcor::Texture> mpFactor;
    Falcor::ref<Falcor::Texture> mpScreenResultTexure;
    Falcor::ref<Falcor::Texture> mpScreenResultWithoutNRC;

    Falcor::ref<Falcor::Buffer> mpSharedCounter;
    uint32_t mTrainDataCount = 0;
    uint32_t mQueryDataCount = 0;
    std::unique_ptr<NRCNetwork::Network> mpNetwork;

    bool isEnabled;
    Falcor::uint mode = viewMode::DEFAULT;

private:
    //void beginFrame(Falcor::RenderContext* pRenderContext);
    //void endFrame(Falcor::RenderContext* pRenderContext);
    
    void registerCudaSource();
    //cudaResource
    struct CudaResource
    {
        NRCData::NRCQueryEntry* queryEntry;
        NRCData::NRCTrainingEntry* trainEntry;
        cudaSurfaceObject_t screenResult;
        ::uint2* queryPixel;
        uint32_t* counterBuffer;
        uint32_t* trainDataCount;
        uint32_t* queryDataCount;
    }mCudaResource;
};
