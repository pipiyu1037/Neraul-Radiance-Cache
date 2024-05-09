from falcor import *

def render_graph_ReSTIRDI():
    g = RenderGraph("ReSTIRDI")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    ReSTIRDI = createPass("ReSTIRDI")
    g.addPass(ReSTIRDI, "ReSTIRDI")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "ReSTIRDI.vbuffer")
    g.addEdge("VBufferRT.mvec", "ReSTIRDI.mvec")
    g.addEdge("ReSTIRDI.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

ReSTIRDI = render_graph_ReSTIRDI()
try: m.addGraph(ReSTIRDI)
except NameError: None