#include "DataStructures.slang"

import Scene.Raytracing;
import Scene.ShadingData;
import Utils.Math.MathHelpers;

struct vertexRecord {
    NRCData::NRCQueryEntry query;
    float3 radiance;
    float pad0;
    float3 thp;
    float pad1;
};

RWStructuredBuffer<NRCData::NRCTrainingEntry> gTrainingData;
RWStructuredBuffer<NRCData::NRCQueryEntry> gNRCQueryData;
RWStructuredBuffer<uint2> gNRCQueryPixel;

RWTexture2D<float4> gScreenResultWithoutNRC;
RWTexture2D<float4> gFactor;

float3 safe_div(float3 a, float3 b)
{
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

NRCData::NRCQueryEntry generateQueryEntry(ShadingData sd, IMaterialInstance mi) {
    NRCData::NRCQueryEntry query = {};
    let BSDFProperties = mi.getProperties(sd);
    query.pos = sd.posW;
    query.dir = cartesian_to_spherical_rad(sd.V);
    query.normal = cartesian_to_spherical_rad(sd.getOrientedFaceNormal());
    query.roughness = 1 - exp(-BSDFProperties.roughness);
    query.diffuse = BSDFProperties.diffuseReflectionAlbedo;
    query.specular = BSDFProperties.specularReflectionAlbedo;
    return query;
}

vertexRecord recordNode(ShadingData sd, IMaterialInstance mi, float3 r, float3 t) {
    NRCData::NRCQueryEntry q = generateQueryEntry(sd, mi);
    vertexRecord record = { q, r, 0.0f, t, 0.0f};
    return record;
}

NRCData::NRCTrainingEntry generateTrainEntry(NRCData::NRCQueryEntry query, float3 radiance, float3 thp) {
    NRCData::NRCTrainingEntry train = {};
    train.query = query;
    train.radiance = radiance;
    train.thp = thp;
    return train;
}
