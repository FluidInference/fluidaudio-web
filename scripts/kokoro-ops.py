# Dump every compute op (Conv/ConvTranspose/MatMul/Gemm/LSTM) with real shapes from
# a profiled ORT run, + ORT kernel-time total and audio length, for the raw-WebGPU
# forward-cost replay (scripts/gpu-kokoro-forward.mjs).
#   cd /tmp/kokoro && python3 kokoro-ops.py   # -> ops.json
import onnxruntime as ort, numpy as np, json, os
so=ort.SessionOptions(); so.enable_profiling=True
sess=ort.InferenceSession('model.onnx', so, providers=['CPUExecutionProvider'])
rng=np.random.default_rng(0); seq=48
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
wav=sess.run(None,{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})[0]
prof=sess.end_profiling(); ev=json.load(open(prof)); os.remove(prof)
ops=[]; tot=0.0
for e in ev:
    if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time'):
        tot+=e['dur']/1e6
        op=e['args'].get('op_name')
        if op not in ('Conv','ConvTranspose','MatMul','Gemm','LSTM'): continue
        a=e['args']
        def shp(key):
            out=[]
            for d in a.get(key,[]):
                out.append(list(d.values())[0])
            return out
        ops.append({'op':op,'in':shp('input_type_shape'),'out':shp('output_type_shape'),'ms':e['dur']/1e6})
json.dump({'audio_s':wav.shape[-1]/24000,'ort_cpu_total_ms':tot,'ops':ops}, open('ops.json','w'))
print(f'audio {wav.shape[-1]/24000:.2f}s  ORT CPU total {tot*1000:.0f}ms  compute ops {len(ops)}')
from collections import Counter
print(Counter(o['op'] for o in ops))
