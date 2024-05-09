#include "NRC.h"

NRC::NRC(Falcor::ref<Falcor::Device> pDevice, unsigned int FrameDimX, unsigned int FrameDimY)
    : mpNetwork(std::make_unique<NRCNetwork::Network>(FrameDimX, FrameDimY))
{
    Falcor::logInfo("Initial NRC...");
    mpDevice = pDevice;
    if (!mpDevice.get()->initCudaDevice())
    {
        FALCOR_THROW("Failed to initialize CUDA device!");
    }

    isEnabled = true;
    mpTrainingData = mpDevice->createStructuredBuffer(
        sizeof(NRCData::NRCTrainingEntry),
        NRCNetwork::max_training_size,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpQueryData = mpDevice->createStructuredBuffer(
        sizeof(NRCData::NRCQueryEntry),
        FrameDimX * FrameDimY,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpQueryPixel = mpDevice->createStructuredBuffer(
        sizeof(Falcor::uint2),
        FrameDimX * FrameDimY,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpSharedCounter = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), 2, Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpScreenResultTexure = mpDevice->createTexture2D(
        FrameDimX,
        FrameDimY,
        Falcor::ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpScreenResultWithoutNRC = mpDevice->createTexture2D(
        FrameDimX,
        FrameDimY,
        Falcor::ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    mpFactor = mpDevice->createTexture2D(
        FrameDimX,
        FrameDimX,
        Falcor::ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess
    );

    if(mpTrainingData == nullptr || mpQueryData == nullptr || mpQueryPixel == nullptr || mpScreenResultTexure == nullptr || mpSharedCounter == nullptr){
        isEnabled = false;
        Falcor::logInfo("failed to init NRC");
    }
    registerCudaSource();
    Falcor::logInfo("Done!");
}

void NRC::registerCudaSource() {
    if (mpScreenResultTexure.get() == nullptr)
    {
        Falcor::logWarning("pScreenResultTexture is Empty!\n");
        exit(-1);
    }
    mCudaResource.screenResult = Falcor::cuda_utils::mapTextureToSurface(mpScreenResultTexure, cudaArrayColorAttachment);
    mCudaResource.queryEntry = (NRCData::NRCQueryEntry*)mpQueryData.get()->getCudaMemory()->getMappedData();
    mCudaResource.trainEntry = (NRCData::NRCTrainingEntry*)mpTrainingData.get()->getCudaMemory()->getMappedData();
    mCudaResource.queryPixel = (uint2*)mpQueryPixel.get()->getCudaMemory()->getMappedData();
    
    mCudaResource.counterBuffer = (uint32_t*)mpSharedCounter->getCudaMemory()->getMappedData();
    mCudaResource.trainDataCount = &mCudaResource.counterBuffer[0];
    mCudaResource.queryDataCount = &mCudaResource.counterBuffer[1];
    Falcor::logInfo("registerCudaResource");
}

bool NRC::renderUI(Falcor::Gui::Widgets& widget) {
    bool optionChanged = false;
    if (isEnabled)
    {
        optionChanged |= widget.dropdown("mode", kNRCViewModeList, mode);
        if (widget.button("reset NRC network"))
        {
            reset();
            optionChanged = true;
        }
    }
    return optionChanged;
}

void NRC::train() {
    // mTrainDataCount = mCudaResource.counterBuffer[0];
    mpNetwork->NRCTrain(mCudaResource.trainEntry, mCudaResource.trainDataCount);
}

void NRC::query() {
    mpNetwork->NRCQuery(mCudaResource.queryEntry, mCudaResource.queryPixel, mQueryDataCount, mCudaResource.screenResult);
}

void NRC::reset() {
    mpNetwork->reset();
}

void NRC::bindShaderData(const Falcor::ShaderVar& var) {
    var["gTrainingData"] = mpTrainingData;
    var["gNRCQueryData"] = mpQueryData;
    var["gNRCQueryPixel"] = mpQueryPixel;
    var["gFactor"] = mpFactor;
    var["gScreenResultWithoutNRC"] = mpScreenResultWithoutNRC;
}
