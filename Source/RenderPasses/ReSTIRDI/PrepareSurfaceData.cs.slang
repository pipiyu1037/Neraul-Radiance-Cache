import LoadShadingData;
import Utils.Color.ColorHelpers;
import Rendering.RTXDI.RTXDI;

/**
 * Prepares RTXDI surface data.
 */
struct PrepareSurfaceData
{
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> texGrads;

    uint2 frameDim;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim))
            return;

        ShadingData sd;
        let lod = ExplicitLodTextureSampler(0.f); 
        bool isValidSurface = loadShadingData(pixel, frameDim, gScene.camera, vbuffer, lod, sd);

        if (isValidSurface)
        {
            // Create material instance and query its properties.
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            let bsdfProperties = mi.getProperties(sd);

            // RTXDI uses a simple material model with only diffuse and specular reflection lobes.
            // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
            // Note: Final shading uses the full material model, the simplified model is only used for resampling purposes.
            gRTXDI.setSurfaceData(
                pixel,
                sd.computeRayOrigin(),
                bsdfProperties.guideNormal,
                bsdfProperties.diffuseReflectionAlbedo,
                bsdfProperties.specularReflectance,
                bsdfProperties.roughness
            );
        }
        else
        {
            gRTXDI.setInvalidSurfaceData(pixel);
        }
    }
};

ParameterBlock<PrepareSurfaceData> gPrepareSurfaceData;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPrepareSurfaceData.execute(dispatchThreadId.xy);
}
