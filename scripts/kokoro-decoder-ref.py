import onnx,numpy as np
from onnx import numpy_helper
g=onnx.load('/tmp/kokoro/model.onnx').graph
I={w.name:numpy_helper.to_array(w).astype(np.float32) for w in g.initializer}
role={}
for n in g.node:
    if n.op_type in ('Conv','Gemm','ConvTranspose') and n.name:
        w=[i for i in n.input if i in I];b=(I[n.input[2]] if len(n.input)>2 and n.input[2] in I else None)
        if w: role[n.name]=(I[w[0]],b,n)
def R(sfx):
    for k in role:
        if k.endswith(sfx):return role[k]
    return None
rd=lambda p:np.frombuffer(open(p,'rb').read(),dtype=np.float32)
sty=rd('/tmp/kokoro/ref_style.bin')[:128]
def lrelu(x):return np.where(x>=0,x,0.2*x)
def conv1d(x,W,b,pad=1,stride=1):
    Co,Ci,K=W.shape;xp=np.pad(x,((0,0),(pad,pad)));T=(xp.shape[1]-K)//stride+1;o=np.zeros((Co,T),np.float32)
    for t in range(T):o[:,t]=W.reshape(Co,-1)@xp[:,t*stride:t*stride+K].reshape(-1)
    return o+(b[:,None] if b is not None else 0)
def dwcT(x,W,b,stride=2,pad=1,opad=1):
    C,_,K=W.shape;L=x.shape[1];Lo=(L-1)*stride-2*pad+K+opad;o=np.zeros((C,Lo),np.float32)
    for t in range(L):
        for k in range(K):
            p=t*stride+k-pad
            if 0<=p<Lo:o[:,p]+=x[:,t]*W[:,0,k]
    return o+(b[:,None] if b is not None else 0)
def adain(x,fc):
    W,b,_=R(fc);h=W@sty+(b if b is not None else 0);C=x.shape[0];g_=h[:C];bt=h[C:]
    mu=x.mean(1,keepdims=True);v=x.var(1,keepdims=True);return (1+g_[:,None])*(x-mu)/np.sqrt(v+1e-5)+bt[:,None]
def block(x,pre,up=False):
    res=x
    if up: res=np.repeat(res,2,axis=1)
    W,b,_=R(f'{pre}/conv1x1/Conv');res=conv1d(res,W,b,pad=0)
    x=lrelu(adain(x,f'{pre}/norm1/fc/Gemm'))
    if up: Wp,bp,_=R(f'{pre}/pool/ConvTranspose');x=dwcT(x,Wp,bp)
    W,b,_=R(f'{pre}/conv1/Conv');x=conv1d(x,W,b)
    x=lrelu(adain(x,f'{pre}/norm2/fc/Gemm'))
    W,b,_=R(f'{pre}/conv2/Conv');x=conv1d(x,W,b)
    return (x+res)/np.sqrt(2)
x=rd('/tmp/kokoro/anc__decoder_decoder_Concat_output_0.bin').reshape(514,15)
asr=rd('/tmp/kokoro/anc__encoder_MatMul_1_output_0.bin').reshape(512,15)
F0=rd('/tmp/kokoro/anc__encoder_F0_proj_Conv_output_0.bin').reshape(1,30)
N=rd('/tmp/kokoro/anc__encoder_N_proj_Conv_output_0.bin').reshape(1,30)
Wa,ba,_=R('asr_res.0/Conv');asr_res=conv1d(asr,Wa,ba,pad=0)
Wf,bf,_=R('F0_conv/Conv');F0d=conv1d(F0,Wf,bf,pad=1,stride=2)
Wn,bn,_=R('N_conv/Conv');Nd=conv1d(N,Wn,bn,pad=1,stride=2)
x=block(x,'encode')
for i,b in enumerate(['decode.0','decode.1','decode.2','decode.3']):
    x=np.concatenate([x,asr_res,F0d,Nd],0)
    x=block(x,b,up=(b=='decode.3'))
    ref=rd(f'/tmp/kokoro/dec__decoder_decoder_{b.replace(".","_")}_Div_{4 if b=="decode.3" else 3}_output_0.bin').reshape(x.shape)
    print(b,"maxΔ",round(float(np.abs(x-ref).max()),4),"shape",x.shape)
np.save('/tmp/kokoro/decode3_out.npy',x)
