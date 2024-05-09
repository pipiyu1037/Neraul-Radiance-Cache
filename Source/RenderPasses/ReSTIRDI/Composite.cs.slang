Texture2D<float4> gNRC;
Texture2D<float4> gScreenResultWithoutNRC;
Texture2D<float4> gFactor;
RWTexture2D<float4> gOutputColor;

enum viewMode
{
    DEFAULT = 0,
    NO_NRC = 1,
    ONLY_QUERY = 2,
    ONLY_FACTOR = 3,
    ONLY_NRC = 4,
};

cbuffer CB
{
    uint mode;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;

    //gOutputColor[pixel] = gNRC[pixel];
    switch(mode){
        case viewMode::DEFAULT:
            gOutputColor[pixel] = gScreenResultWithoutNRC[pixel] + gNRC[pixel] * gFactor[pixel];
            break;
        case viewMode::NO_NRC:
            gOutputColor[pixel] = gScreenResultWithoutNRC[pixel];
            break;
        case viewMode::ONLY_QUERY:
            gOutputColor[pixel] = gNRC[pixel];
            break;
        case viewMode::ONLY_FACTOR:
            gOutputColor[pixel] = gFactor[pixel];
            break;
        case viewMode::ONLY_NRC:
            gOutputColor[pixel] = gNRC[pixel] * gFactor[pixel];
            break;
        default:
            gOutputColor[pixel] = gScreenResultWithoutNRC[pixel] + gNRC[pixel] * gFactor[pixel];
    }
}
