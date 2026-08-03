# Profile a standalone encoder ONNX: dump compute-op shapes + confirm M sizes.
import onnxruntime as ort, numpy as np, json, sys, os
from collections import defaultdict, Counter
path=sys.argv[1]; T=int(sys.argv[2]) if len(sys.argv)>2 else 1500
so=ort.SessionOptions(); so.enable_profiling=True
sess=ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
feeds={}
for i in sess.get_inputs():
    sh=[d if isinstance(d,int) and d>0 else (T if ('freq' not in i.name.lower()) else 128) for d in i.shape]
    # heuristics for parakeet encoder: audio_signal[1,128,T], length[1]
    if 'length' in i.name or i.name=='length': feeds[i.name]=np.array([T],dtype=np.int64)
    else:
        sh=[1,128,T] if len(i.shape)==3 else ([1] if len(i.shape)==1 else [max(1,x) for x in sh])
        feeds[i.name]=np.random.randn(*sh).astype(np.float32)
print('inputs:',{k:list(v.shape) for k,v in feeds.items()})
import time;t0=time.time()
outs=sess.run(None,feeds); dt=time.time()-t0
prof=sess.end_profiling(); ev=json.load(open(prof)); os.remove(prof)
ops=[]; tot=0; byop=defaultdict(float); Ms=[]
for e in ev:
    if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time'):
        tot+=e['dur']/1e6; op=e['args'].get('op_name'); byop[op]+=e['dur']/1e6
        if op not in ('Conv','ConvTranspose','MatMul','FusedMatMul','Gemm','LSTM','FusedConv'): continue
        a=e['args']
        def shp(k):return [list(d.values())[0] for d in a.get(k,[])]
        ins=shp('input_type_shape'); out=shp('output_type_shape')
        ops.append({'op':op.replace('Fused',''),'in':ins,'out':out,'ms':e['dur']/1e6})
        if op in ('MatMul','FusedMatMul','Gemm') and ins and len(ins[0])>=2:
            Ms.append(int(np.prod(ins[0][:-1])))
json.dump({'ort_cpu_ms':tot*1000,'ops':ops},open('/tmp/pkv3/enc-ops.json','w'))
print(f'ORT CPU encoder {tot*1000:.0f}ms  compute ops {len(ops)}')
print('top ops(ms):',{k:round(v*1000) for k,v in sorted(byop.items(),key=lambda x:-x[1])[:6]})
print(f'MatMul M dims: min {min(Ms) if Ms else 0} max {max(Ms) if Ms else 0} median {int(np.median(Ms)) if Ms else 0}  (Kokoro was ~128; large M => f16 helps)')
