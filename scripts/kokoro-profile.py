import onnxruntime as ort, numpy as np, json, glob, os
from collections import defaultdict
so=ort.SessionOptions(); so.enable_profiling=True
sess=ort.InferenceSession('model.onnx', so, providers=['CPUExecutionProvider'])
rng=np.random.default_rng(0); seq=48
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
import time; t0=time.time()
wav=sess.run(None,{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})[0]
dt=time.time()-t0
prof=sess.end_profiling()
ev=json.load(open(prof))
byop=defaultdict(float); bymod=defaultdict(float); tot=0
for e in ev:
    if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time'):
        d=e['dur']/1e6; tot+=d
        byop[e['args'].get('op_name','?')]+=d
        nm=e.get('name','')
        mod='decoder' if '/decoder' in nm else ('predictor' if 'predictor' in nm else ('bert' if 'bert' in nm else ('text_encoder' if 'text_encoder' in nm else 'other')))
        bymod[mod]+=d
audio_s=wav.shape[-1]/24000
print(f'audio {audio_s:.2f}s  seq {seq}  wall {dt*1000:.0f}ms  kernel_sum {tot*1000:.0f}ms  RTFx(CPU) {audio_s/dt:.2f}')
print('by op (top):'); 
for k,v in sorted(byop.items(),key=lambda x:-x[1])[:8]: print(f'  {k:16s} {v*1000:7.1f} ms  {100*v/tot:4.1f}%')
print('by module:')
for k,v in sorted(bymod.items(),key=lambda x:-x[1]): print(f'  {k:14s} {v*1000:7.1f} ms  {100*v/tot:4.1f}%')
os.remove(prof)
