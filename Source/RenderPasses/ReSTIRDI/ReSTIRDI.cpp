/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReSTIRDI.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

using namespace Falcor;

namespace
{
const std::string kPrepareSurfaceDataFile = "RenderPasses/ReSTIRDI/PrepareSurfaceData.cs.slang";
const std::string kFinalShadingFile = "RenderPasses/ReSTIRDI/FinalShading.cs.slang";
const std::string kPathTracerFile = "RenderPasses/ReSTIRDI/PathTracer.rt.slang";
const std::string kCompositeFile = "RenderPasses/ReSTIRDI/Composite.cs.slang";

const std::string kInputVBuffer = "vbuffer";
const std::string kInputTexGrads = "texGrads";
const std::string kInputMotionVectors = "mvec";
const std::string kInputViewDir = "viewW";

const std::string kOutputReSTIRDI = "OuputReSTIRDI";

const uint32_t kMaxPayloadSizeBytes = 512u;
const uint32_t kMaxRecursionDepth = 4u;

const Falcor::ChannelList kInputChannels = {
    // clang-format off
    { kInputVBuffer,            "gVBuffer",                 "Visibility buffer in packed format"                       },
    //{ kInputTexGrads,           "gTextureGrads",            "Texture gradients", true /* optional */                   },
    { kInputMotionVectors,      "gMotionVector",            "Motion vector buffer (float format)", true /* optional */ },
    { kInputViewDir,            "gViewW",                   "World-space view direction (xyz float format)", true /* optional */},
    // clang-format on
};

const Falcor::ChannelList kOutputChannels = {
    // clang-format off
    { "color",                  "gOutputColor",                   "Final color",              true /* optional */, ResourceFormat::RGBA32Float },
    { kOutputReSTIRDI,          "gOutputReSTIRDI",                "ReSTIRDI color",           true /* optional */, ResourceFormat::RGBA32Float },
    { "emission",               "gEmission",                "Emissive color",           true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseIllumination",    "gDiffuseIllumination",     "Diffuse illumination",     true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseReflectance",     "gDiffuseReflectance",      "Diffuse reflectance",      true /* optional */, ResourceFormat::RGBA32Float },
    { "specularIllumination",   "gSpecularIllumination",    "Specular illumination",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specularReflectance",    "gSpecularReflectance",     "Specular reflectance",     true /* optional */, ResourceFormat::RGBA32Float },
    //{ "trainPixel","gTrainPixel", "train data pixel", true, ResourceFormat::RGBA32Float},
    //{ "trainLength", "gTrainLength", "train path Length", true, ResourceFormat::RGBA32Float},
    //{ "colorWithoutNRC", "gScreenResultWithoutNRC", "color without NRC", true, ResourceFormat::RGBA32Float},
    //{ "NRCScreenResult", "gNRCScreenResult", "NRC query Result", true, ResourceFormat::RGBA32Float},
    // clang-format on
};

const Falcor::ChannelList kReSTIRDIOutputChannels = {
    // clang-format off
    { "color",                  "gOutputColor",                   "Final color",              true /* optional */, ResourceFormat::RGBA32Float },
    { "emission",               "gEmission",                "Emissive color",           true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseIllumination",    "gDiffuseIllumination",     "Diffuse illumination",     true /* optional */, ResourceFormat::RGBA32Float },
    { "diffuseReflectance",     "gDiffuseReflectance",      "Diffuse reflectance",      true /* optional */, ResourceFormat::RGBA32Float },
    { "specularIllumination",   "gSpecularIllumination",    "Specular illumination",    true /* optional */, ResourceFormat::RGBA32Float },
    { "specularReflectance",    "gSpecularReflectance",     "Specular reflectance",     true /* optional */, ResourceFormat::RGBA32Float },
    // clang-format on
};

// Scripting options.
const char* kOptions = "options";
const char* kMaxBounces = "maxBounces";
const char* kComputeDirect = "computeDirect";
const char* kUseImportanceSampling = "useImportanceSampling";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRDI>();
}

ReSTIRDI::ReSTIRDI(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void ReSTIRDI::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kOptions)
            mOptions = value;
        else if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
        {
            logWarning("Unknown property '{}' in ReSTIRDI properties.", key);
        }
    }
}

RenderPassReflection ReSTIRDI::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    
    return reflector;
}

void ReSTIRDI::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
    {
        clearRenderPassChannels(pRenderContext, kOutputChannels, renderData);
        return;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        recreatePrograms();
    }
    
    FALCOR_ASSERT(mpRTXDI);

    if (!mpTracer)
        mpTracer = std::make_unique<Tracer>(mpDevice, mpScene);

    const auto& pVBuffer = renderData.getTexture(kInputVBuffer);
    const auto& pMotionVectors = renderData.getTexture(kInputMotionVectors);

    auto& dict = renderData.getDictionary();

    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    mGBufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);

    mpRTXDI->beginFrame(pRenderContext, mFrameDim);

    prepareSurfaceData(pRenderContext, pVBuffer);

    mpRTXDI->update(pRenderContext, pMotionVectors);

    // finalShading(pRenderContext, pVBuffer, renderData);

    pathTrace(pRenderContext, renderData, *mpTracer);

    mpRTXDI->endFrame(pRenderContext);

    composite(pRenderContext, renderData);
}

void ReSTIRDI::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpRTXDI = nullptr;
    mNRC = nullptr;
    recreatePrograms();

    if (mpScene)
    {
        if (pScene->hasProceduralGeometry())
        {
            logWarning("ReSTIRDI: This render pass only supports triangles. Other types of geometry will be ignored.");
        }
        mpRTXDI = std::make_unique<RTXDI>(mpScene, mOptions);
    }
}

bool ReSTIRDI::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpRTXDI ? mpRTXDI->getPixelDebug().onMouseEvent(mouseEvent) : false;
}

Properties ReSTIRDI::getProperties() const
{
    Properties props;
    props[kOptions] = mOptions;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

void ReSTIRDI::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void ReSTIRDI::renderUI(Gui::Widgets& widget)
{
    if (mpRTXDI)
    {
        mOptionsChanged = mpRTXDI->renderUI(widget);
        if (mOptionsChanged)
            mOptions = mpRTXDI->getOptions();
    }
    if (mNRC)
    {
        mOptionsChanged |= mNRC->renderUI(widget);
    }
}

void ReSTIRDI::recreatePrograms() {
    mpPrepareSurfaceDataPass = nullptr;
    mpFinalShadingPass = nullptr;
    mpTracer = nullptr;
    mFrameCount = 0;
}

void ReSTIRDI::prepareVars() {
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mpTracer->pProgram);

    // Configure program.
    mpTracer->pProgram->addDefines(mpSampleGenerator->getDefines());
    mpTracer->pProgram->setTypeConformances(mpScene->getTypeConformances());
    
    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mpTracer->pVars = RtProgramVars::create(mpDevice, mpTracer->pProgram, mpTracer->pBindingTable);

    // Bind utility classes into shared data.
    auto var = mpTracer->pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}

ReSTIRDI::Tracer::Tracer(ref<Device> pDevice, ref<Scene> pScene)
{
    if (pScene->hasProceduralGeometry())
    {
        logWarning("ReSTIRPass: This render pass only supports triangles. Other types of geometry will be ignored.");
    }

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kPathTracerFile);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

    pBindingTable = RtBindingTable::create(2, 2, pScene->getGeometryCount());
    auto& sbt = pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("scatterMiss"));
    sbt->setMiss(1, desc.addMiss("shadowMiss"));

    if (pScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
    {
        sbt->setHitGroup(
            0,
            pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
            desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
        );
        sbt->setHitGroup(1, pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit"));
    }

    if (pScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
    {
        sbt->setHitGroup(
            0,
            pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
            desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
        );
        sbt->setHitGroup(
            1,
            pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
            desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
        );
    }

    if (pScene->hasGeometryType(Scene::GeometryType::Curve))
    {
        sbt->setHitGroup(
            0, pScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
        );
        sbt->setHitGroup(1, pScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
    }

    if (pScene->hasGeometryType(Scene::GeometryType::SDFGrid))
    {
        sbt->setHitGroup(
            0,
            pScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
            desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
        );
        sbt->setHitGroup(1, pScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
    }

    pProgram = Program::create(pDevice, desc, pScene->getSceneDefines());
}

void ReSTIRDI::prepareSurfaceData(RenderContext* pRenderContext, const ref<Texture>& pVBuffer) {
    FALCOR_ASSERT(mpRTXDI);
    FALCOR_ASSERT(pVBuffer);

    FALCOR_PROFILE(pRenderContext, "prepareSurfaceData");

    if (!mpPrepareSurfaceDataPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPrepareSurfaceDataFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        defines.add(mpRTXDI->getDefines());
        defines.add("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

        mpPrepareSurfaceDataPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mpPrepareSurfaceDataPass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

    auto rootVar = mpPrepareSurfaceDataPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpRTXDI->bindShaderData(rootVar);

    auto var = rootVar["gPrepareSurfaceData"];
    var["vbuffer"] = pVBuffer;
    var["frameDim"] = mFrameDim;

    mpPrepareSurfaceDataPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ReSTIRDI::finalShading(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const RenderData& renderData) {
    FALCOR_ASSERT(mpRTXDI);
    FALCOR_ASSERT(pVBuffer);

    FALCOR_PROFILE(pRenderContext, "finalShading");

    if (!mpFinalShadingPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kFinalShadingFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        defines.add(mpRTXDI->getDefines());
        defines.add("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
        defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
        defines.add(getValidResourceDefines(kOutputChannels, renderData));

        mpFinalShadingPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mpFinalShadingPass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
    mpFinalShadingPass->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    mpFinalShadingPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    auto rootVar = mpFinalShadingPass->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);
    mpRTXDI->bindShaderData(rootVar);

    auto var = rootVar["gFinalShading"];
    var["vbuffer"] = pVBuffer;
    var["frameDim"] = mFrameDim;

    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = renderData.getTexture(channel.name);
        rootVar[channel.texname] = pTex;
    };

    for (const auto& channel : kReSTIRDIOutputChannels)
        bind(channel);

    mpFinalShadingPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ReSTIRDI::pathTrace(RenderContext* pRenderContext, const RenderData& renderData, Tracer& mTracer)
{
    FALCOR_ASSERT(mpRTXDI);
    FALCOR_PROFILE(pRenderContext, "pathTracer");
    const Falcor::uint2 targetDim = renderData.getDefaultTextureDims();

    if (mFrameCount == 0)
    {
        mNRC = std::make_unique<NRC>(mpDevice, targetDim.x, targetDim.y);
    }

    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    mTracer.pProgram->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");

    if (mNRC != nullptr)
    {
        mTracer.pProgram->addDefine("USE_NRC", mNRC->isEnabled ? "1" : "0");
    }

    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    mTracer.pProgram->addDefines(mpRTXDI->getDefines());

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    auto& dict = renderData.getDictionary();

    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    
    mpScene->bindShaderData(var["gScene"]);
    if (mpRTXDI)
        mpRTXDI->bindShaderData(var);

    auto bind = [&](const ChannelDesc& channel)
    {
        ref<Texture> pTex = renderData.getTexture(channel.name);
        var[channel.texname] = pTex;
    };

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);
    var["gViewW"] = renderData.getTexture(kInputViewDir);

    for (const auto& channel : kOutputChannels)
        bind(channel);

    if (mNRC)
    {
        pRenderContext->clearUAVCounter(mNRC->mpQueryData, 0);
        pRenderContext->clearUAVCounter(mNRC->mpTrainingData, 0);
        pRenderContext->clearUAVCounter(mNRC->mpQueryPixel, 0);
        mNRC->bindShaderData(var);
    }

    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, Falcor::uint3(targetDim, 1));
    mFrameCount++;
}

void ReSTIRDI::composite(RenderContext* pRenderContext, const RenderData& renderData) {
    if (mNRC == nullptr)
        return;

    FALCOR_PROFILE(pRenderContext, "composite");

    pRenderContext->copyBufferRegion(mNRC->mpSharedCounter.get(), 0, mNRC->mpTrainingData->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mNRC->mpSharedCounter.get(), 4, mNRC->mpQueryData->getUAVCounter().get(), 0, 4);

    mNRC->mTrainDataCount = mNRC->mpSharedCounter->getElement<uint32_t>(0);
    mNRC->mQueryDataCount = mNRC->mpSharedCounter->getElement<uint32_t>(1);
    
    mNRC->train();
    cudaDeviceSynchronize();
    mNRC->query();
    cudaDeviceSynchronize();

    if (!mpCompositePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kCompositeFile).csEntry("main");
        mpCompositePass = ComputePass::create(mpDevice, desc);
    }

    auto vars = mpCompositePass->getRootVar();

    vars["gNRC"] = mNRC->mpScreenResultTexure;
    vars["gScreenResultWithoutNRC"] = mNRC->mpScreenResultWithoutNRC;
    vars["gFactor"] = mNRC->mpFactor;
    vars["gOutputColor"] = renderData.getTexture("color");
    vars["CB"]["mode"] = mNRC->mode;

    mpCompositePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}
