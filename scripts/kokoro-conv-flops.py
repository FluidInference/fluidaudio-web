import onnxruntime as ort, numpy as np, json, os
from collections import defaultdict
so=ort.SessionOptions(); so.enable_profiling=True
sess=ort.InferenceSession('model.onnx', so, providers=['CPUExecutionProvider'])
rng=np.random.default_rng(0); seq=48
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
wav=sess.run(None,{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})[0]
prof=sess.end_profiling(); ev=json.load(open(prof)); os.remove(prof)
# profiling events carry input/output shapes in args
convs=[]
for e in ev:
    if e.get('cat')=='Node' and e.get('name','').endswith('_kernel_time') and e['args'].get('op_name') in ('Conv','ConvTranspose'):
        a=e['args']
        ins=a.get('input_type_shape',[]); outs=a.get('output_type_shape',[])
        convs.append((e['args']['op_name'], e['dur']/1e6, ins, outs))
tot_fl=0; big=[]
for op,dur,ins,outs in convs:
    try:
        # ins[0]=activation [N,Cin,L], ins[1]=weight [Cout,Cin/g,K] or [Cin,Cout/g,K]
        act=list(ins[0].values())[0]; w=list(ins[1].values())[0]; out=list(outs[0].values())[0]
        Cin=act[1]; Lout=out[-1]; Cout=out[1]; K=w[-1]; Cing=w[1]
        fl=2*Cout*Cing*K*Lout/1e9
        tot_fl+=fl; big.append((op,fl,dur*1000,act,w,out))
    except Exception as ex: pass
big.sort(key=lambda x:-x[1])
print(f'audio {wav.shape[-1]/24000:.2f}s  total conv GFLOPs {tot_fl:.2f}')
print('heaviest convs (GFLOP, ms, act->out):')
for op,fl,ms,act,w,out in big[:8]:
    print(f'  {op:14s} {fl:5.2f}G {ms:6.1f}ms  in{act} w{w} out{out}')
