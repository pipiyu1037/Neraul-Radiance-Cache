add_plugin(ReSTIRDI)

target_sources(ReSTIRDI PRIVATE
    ReSTIRDI.cpp
    ReSTIRDI.h
    NRC.cpp
    NRC.h
    
    Network.cu
    Network.h

    FinalShading.cs.slang
    LoadShadingData.slang
    PrepareSurfaceData.cs.slang
    PathTracer.rt.slang
    DataStructures.slang
    NRC.slang
    Composite.cs.slang
)
set_target_properties(ReSTIRDI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ReSTIRDI PUBLIC ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

target_copy_shaders(ReSTIRDI RenderPasses/ReSTIRDI)

target_source_group(ReSTIRDI "RenderPasses")
