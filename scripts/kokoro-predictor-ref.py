import onnx, numpy as np
from onnx import numpy_helper
g = onnx.load('/tmp/kokoro/model.onnx').graph
I = {w.name: numpy_helper.to_array(w).astype(np.float32) for w in g.initializer}
lstm_io = {}   # node-name-suffix -> (W,R,B)
gemm = {}
for n in g.node:
    if n.op_type == 'LSTM':
        lstm_io[n.name] = (I[n.input[1]], I[n.input[2]], I[n.input[3]])
    if n.op_type in ('Gemm', 'MatMul') and n.name:
        w = [i for i in n.input if i in I]
        if w: gemm[n.name] = (I[w[0]], I[n.input[2]] if len(n.input) > 2 and n.input[2] in I else None)
def L(sfx):
    return next(v for k, v in lstm_io.items() if k.endswith(sfx))
def G(sfx):
    return next(v for k, v in gemm.items() if k.endswith(sfx))
rd = lambda p: np.frombuffer(open(p, 'rb').read(), dtype=np.float32)

sig = lambda x: 1/(1+np.exp(-x))
def lstm_bidir(x, W, R, B, H):
    # x [seq,inp]; ONNX iofc; returns [seq,2H]
    seq = x.shape[0]; out = np.zeros((seq, 2*H), np.float32)
    for d in range(2):
        Wd, Rd, Bd = W[d], R[d], B[d]  # [4H,inp],[4H,H],[8H]
        h = np.zeros(H, np.float32); c = np.zeros(H, np.float32)
        order = range(seq) if d == 0 else range(seq-1, -1, -1)
        for t in order:
            z = Wd @ x[t] + Rd @ h + Bd[:4*H] + Bd[4*H:]
            i, o, f, cc = sig(z[:H]), sig(z[H:2*H]), sig(z[2*H:3*H]), np.tanh(z[3*H:])
            c = f*c + i*cc; h = o*np.tanh(c)
            out[t, d*H:(d+1)*H] = h
    return out
def layernorm(x, g_, b_, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x-m)/np.sqrt(v+eps)*g_+b_

d_en = rd('/tmp/kokoro/anc__encoder_bert_encoder_Add_output_0.bin')
style = rd('/tmp/kokoro/ref_style.bin')
sp = style[128:256]   # prosodic half
seq = d_en.size // 512
d_en = d_en.reshape(seq, 512)
print("seq", seq)

# DurationEncoder: 3 (bidir-LSTM, AdaLN) pairs. input concat(x, sp)
x = d_en
for i, (li, ai) in enumerate([(0,1),(2,3),(4,5)]):
    W, R, B = L(f'text_encoder/lstms.{li}/LSTM')
    xin = np.concatenate([x, np.tile(sp, (seq,1))], 1)
    x = lstm_bidir(xin, W, R, B, 256)   # [seq,512]
    gw, gb = G(f'text_encoder/lstms.{ai}/fc/Gemm')  # [1024,128]
    h = gw @ sp + (gb if gb is not None else 0)      # [1024]
    g_, b_ = h[:512], h[512:]
    x = layernorm(x, 1+g_, b_)
dur_enc_out = x  # [seq,512] BEFORE predictor.lstm (used for alignment d)
# predictor.lstm
W, R, B = L('predictor/lstm/LSTM')
xin = np.concatenate([x, np.tile(sp, (seq,1))], 1)
xp = lstm_bidir(xin, W, R, B, 256)  # [seq,512]
# duration_proj
dw, db = G('duration_proj/linear_layer/MatMul')  # [512,50]
# find the bias init
dbias = I.get('encoder.predictor.duration_proj.linear_layer.bias')
dur = sig(xp @ dw + (dbias if dbias is not None else 0))  # [seq,50]
dur = dur.sum(1)          # [seq]
pred = np.clip(np.round(dur), 1, 50).astype(np.int32)

ref = rd('/tmp/kokoro/anc__encoder_Clip_output_0.bin')
print("pred_dur match", int((pred == np.round(ref).astype(int)).all()), "sum(T)", int(pred.sum()))
predR = np.round(ref).astype(int)  # use ORT dur for alignment

# ── alignment: A[seq,T], en = d^T @ A ──
T = int(predR.sum())
A = np.zeros((seq, T), np.float32); f = 0
for i in range(seq):
    A[i, f:f+predR[i]] = 1.0; f += predR[i]
d = np.concatenate([dur_enc_out, np.tile(sp, (seq,1))], 1)  # [seq,640]
en = d.T @ A  # [640,T]
enRef = rd('/tmp/kokoro/anc__encoder_MatMul_output_0.bin').reshape(640, T)
print("en maxΔ", float(np.abs(en-enRef).max()))

# ── shared LSTM → prosody [512,T] ──
W, R, B = L('shared/LSTM')
sh = lstm_bidir(en.T, W, R, B, 256)  # [T,512]
prosody = sh.T  # [512,T]

# ── F0/N AdaINResBlocks (prosodic style) ──
def conv1d_np(x, W, b, pad=1, stride=1):
    Co, Ci, K = W.shape; xp = np.pad(x, ((0,0),(pad,pad))); Tn=(xp.shape[1]-K)//stride+1
    o = np.zeros((Co,Tn),np.float32)
    for t in range(Tn): o[:,t]=W.reshape(Co,-1)@xp[:,t*stride:t*stride+K].reshape(-1)
    return o+(b[:,None] if b is not None else 0)
def dwcT_np(x, W, b, stride=2, pad=1, opad=1):
    C,_,K=W.shape; Ln=x.shape[1]; Lo=(Ln-1)*stride-2*pad+K+opad; o=np.zeros((C,Lo),np.float32)
    for t in range(Ln):
        for k in range(K):
            p=t*stride+k-pad
            if 0<=p<Lo: o[:,p]+=x[:,t]*W[:,0,k]
    return o+(b[:,None] if b is not None else 0)
def adain_np(x, sfx):
    gw,gb=G(f'{sfx}/fc/Gemm'); h=gw@sp+(gb if gb is not None else 0); C=x.shape[0]
    mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True)
    return (1+h[:C,None])*(x-mu)/np.sqrt(v+1e-5)+h[C:,None]
lrelu=lambda x:np.where(x>=0,x,0.2*x)
convrole={}; ctrole={}
for n in g.node:
    if n.op_type in ('Conv','ConvTranspose') and n.name:
        w=[i for i in n.input if i in I]
        if w:
            role=(I[w[0]], I[n.input[2]] if len(n.input)>2 and n.input[2] in I else None)
            (ctrole if n.op_type=='ConvTranspose' else convrole)[n.name]=role
def C_(sfx): return next(v for k,v in convrole.items() if k.endswith(sfx))
def CT_(sfx): return next((v for k,v in ctrole.items() if k.endswith(sfx)), None)
def block_pred(x, pre, up):
    res=x
    if up: res=np.repeat(res,2,axis=1)
    try: Wc,bc=C_(f'{pre}/conv1x1/Conv'); res=conv1d_np(res,Wc,bc,pad=0)
    except StopIteration: pass  # no channel change → identity residual
    x=lrelu(adain_np(x,f'{pre}/norm1'))
    if up:
        p=CT_(f'{pre}/pool/ConvTranspose')
        if p is not None: x=dwcT_np(x,p[0],p[1])
    Wc,bc=C_(f'{pre}/conv1/Conv'); x=conv1d_np(x,Wc,bc,pad=1)
    x=lrelu(adain_np(x,f'{pre}/norm2'))
    Wc,bc=C_(f'{pre}/conv2/Conv'); x=conv1d_np(x,Wc,bc,pad=1)
    return (x+res)/np.sqrt(2)
def proj(x, sfx):
    Wc,bc=C_(sfx); return conv1d_np(x,Wc,bc,pad=0 if Wc.shape[2]==1 else 1)
xf=prosody
for b in ['F0.0','F0.1','F0.2']: xf=block_pred(xf,b,up=(b=='F0.1'))
F0=proj(xf,'F0_proj/Conv')
xn=prosody
for b in ['N.0','N.1','N.2']: xn=block_pred(xn,b,up=(b=='N.1'))
Nv=proj(xn,'N_proj/Conv')
F0ref=rd('/tmp/kokoro/anc__encoder_F0_proj_Conv_output_0.bin')
Nref=rd('/tmp/kokoro/anc__encoder_N_proj_Conv_output_0.bin')
print("F0 shape",F0.shape,"maxΔ",float(np.abs(F0.ravel()-F0ref).max()))
print("N  shape",Nv.shape,"maxΔ",float(np.abs(Nv.ravel()-Nref).max()))

# ── Chain A: encoder.text_encoder (embedding → 3× [Conv k5p2, LN, LeakyRelu] → LSTM) → align ──
ids=np.frombuffer(open('/tmp/kokoro/ref_ids.bin','rb').read(),dtype=np.int32)
emb=I['encoder.text_encoder.embedding.weight'][ids]  # [seq,512]
def LN_C(x,gw,gb,eps=1e-5):  # x[C,L] LayerNorm over C per column
    m=x.mean(0,keepdims=True); v=x.var(0,keepdims=True); return (x-m)/np.sqrt(v+eps)*gw[:,None]+gb[:,None]
lnw={};
for n in g.node:
    if n.op_type=='LayerNormalization' and n.name:
        ws=[i for i in n.input if i in I];
        if len(ws)>=2: lnw[n.name]=(I[ws[0]],I[ws[1]])
def LNW(sfx): return next(v for k,v in lnw.items() if k.endswith(sfx))
xe=emb.T  # [512,seq]
for ci,c in enumerate(['cnn.0','cnn.1','cnn.2']):
    Wc,bc=C_(f'{c}/cnn.{c[-1]}.0/Conv'); xe=conv1d_np(xe,Wc,bc,pad=2)
    if ci==0: np.save('/tmp/kokoro/dbg_te_conv0.npy', xe)
    gw,gb=LNW(f'{c}/cnn.{c[-1]}.1/LayerNormalization'); xe=LN_C(xe,gw,gb)
    if ci==0: np.save('/tmp/kokoro/dbg_te_ln0.npy', xe)
    xe=lrelu(xe)
np.save('/tmp/kokoro/dbg_te_cnn.npy', xe)  # [512,seq] after CNN
te_lstm=lstm_bidir(xe.T,*L('encoder/text_encoder/lstm/LSTM'),256)  # [seq,512]
np.save('/tmp/kokoro/dbg_te_lstm.npy', te_lstm)
asr=te_lstm.T @ A  # [512,T]
asrRef=rd('/tmp/kokoro/anc__encoder_MatMul_1_output_0.bin').reshape(512,T)
print("asr shape",asr.shape,"maxΔ",float(np.abs(asr-asrRef).max()))
# decoder input Concat = [asr; F0_conv(F0) s2; N_conv(N) s2] = [514,T]
Wf,bf=C_('F0_conv/Conv'); F0d=conv1d_np(F0,Wf,bf,pad=1,stride=2)
Wn,bn=C_('N_conv/Conv');  Nd=conv1d_np(Nv,Wn,bn,pad=1,stride=2)
concat=np.concatenate([asr,F0d,Nd],0)
cref=rd('/tmp/kokoro/anc__decoder_decoder_Concat_output_0.bin').reshape(514,T)
print("Concat[514,T] maxΔ",float(np.abs(concat-cref).max()))

# ── SineGen (m_source): F0[1,30] → source waveform[9000], gate vs msrc_tanh ──
sr=24000.0; harm=np.arange(1,10,dtype=np.float32)
lw=I['decoder.decoder.generator.m_source.l_linear.weight'] if 'decoder.decoder.generator.m_source.l_linear.weight' in I else None
# l_linear weight [9,1]
lw=next(I[i] for n in g.node if 'm_source' in n.name and n.op_type=='MatMul' for i in n.input if i in I)
lb=I['decoder.decoder.generator.m_source.l_linear.bias']
F0src=F0.ravel()  # [30]
Fs=30  # F0 frames
F0up=np.repeat(F0src,300)  # [9000] nearest upsample x300
uv=(F0up>10).astype(np.float32)
fh=F0up[:,None]*harm                       # [9000,9]
frac=fh/sr; frac=frac-np.floor(frac)       # [9000,9]
fr_d=frac[::300]                           # downsample x1/300 -> [30,9]
ref=rd('/tmp/kokoro/msrc_tanh.bin')
def lin_up(ph, factor):  # linear interpolate along axis0 by factor (align_corners=False like onnx)
    n=ph.shape[0]; out=np.zeros((n*factor,ph.shape[1]),np.float32)
    for j in range(n*factor):
        pos=(j+0.5)/factor-0.5; lo=int(np.floor(pos)); hi=min(lo+1,n-1); lo=max(lo,0); w=pos-np.floor(pos)
        out[j]=ph[lo]*(1-w)+ph[hi]*w
    return out
for scale in [1.0,300.0]:
    for up in ['rep','lin']:
        ph=np.cumsum(fr_d*scale,axis=0)*2*np.pi
        ph_u=np.repeat(ph,300,axis=0) if up=='rep' else lin_up(ph,300)
        sines=np.sin(ph_u)*0.1*uv[:len(ph_u),None]
        src=np.tanh(sines@lw+lb).ravel()
        m=min(len(src),len(ref)); c=np.corrcoef(src[:m],ref[:m])[0,1]
        print(f"SineGen scale={scale} up={up}: corr {c:.4f} maxΔ {np.abs(src[:m]-ref[:m]).max():.4f}")
