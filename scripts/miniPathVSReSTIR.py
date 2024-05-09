from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_miniPathVSReSTIR():
    g = RenderGraph('miniPathVSReSTIR')
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ReSTIRDI', 'ReSTIRDI', {'options': {'mode': 'SpatiotemporalResampling', 'presampledTileCount': 128, 'presampledTileSize': 1024, 'storeCompactLightInfo': True, 'localLightCandidateCount': 24, 'infiniteLightCandidateCount': 8, 'envLightCandidateCount': 8, 'brdfCandidateCount': 1, 'brdfCutoff': 0.0, 'testCandidateVisibility': True, 'biasCorrection': 'Basic', 'depthThreshold': 0.10000000149011612, 'normalThreshold': 0.5, 'samplingRadius': 30.0, 'spatialSampleCount': 1, 'spatialIterations': 5, 'maxHistoryLength': 20, 'boilingFilterStrength': 0.0, 'rayEpsilon': 0.0010000000474974513, 'useEmissiveTextures': False, 'enableVisibilityShortcut': False, 'enablePermutationSampling': False}, 'maxBounces': 3, 'computeDirect': True, 'useImportanceSampling': True})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('SplitScreenPass', 'SplitScreenPass', {'splitLocation': 0.5009999871253967, 'showTextLabels': True, 'leftLabel': 'ReSTIR', 'rightLabel': 'miniPathTracer'})
    g.create_pass('MinimalPathTracer', 'MinimalPathTracer', {'maxBounces': 3, 'computeDirect': True, 'useImportanceSampling': True})
    g.create_pass('AccumulatePass0', 'AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('ToneMapper0', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.add_edge('VBufferRT.vbuffer', 'ReSTIRDI.vbuffer')
    g.add_edge('VBufferRT.mvec', 'ReSTIRDI.mvec')
    g.add_edge('ReSTIRDI.color', 'AccumulatePass.input')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('VBufferRT.vbuffer', 'MinimalPathTracer.vbuffer')
    g.add_edge('VBufferRT.viewW', 'MinimalPathTracer.viewW')
    g.add_edge('MinimalPathTracer.color', 'AccumulatePass0.input')
    g.add_edge('AccumulatePass0.output', 'ToneMapper0.src')
    g.add_edge('ToneMapper0.dst', 'SplitScreenPass.rightInput')
    g.add_edge('ToneMapper.dst', 'SplitScreenPass.leftInput')
    g.mark_output('SplitScreenPass.output')
    return g

miniPathVSReSTIR = render_graph_miniPathVSReSTIR()
try: m.addGraph(miniPathVSReSTIR)
except NameError: None
