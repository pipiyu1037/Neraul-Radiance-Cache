#pragma once

namespace NRCData {

struct NRCQueryEntry {
    float3 pos;
    float roughness;
    float2 dir;
    float2 normal; 
    float3 specular;
    float pad0 = 1.0f;
    float3 diffuse;
    float pad1 = 1.0f;

    /*__init(){
        this.pos = float3(0.0f);
        this.pad0 = 1.0f;
        this.dir = float2(0.0f);
        this.normal = float2(0.0f);
        this.roughness = 0.0f;
        this.specular = float3(0.0f);
        this.diffuse = float3(0.0f);
        this.pad1 = 1.0f;
    }*/
};

struct NRCTrainingEntry {
    NRCQueryEntry query;
    float3 radiance;
    uint32_t idx = 0;
    float3 thp;
    float pad;
    
    /*__init() {
        this.query = NRCData::NRCQueryEntry();
        this.radiance = float3(0.0f);
        this.thp = float3(0.0f);
    }*/
};

}
