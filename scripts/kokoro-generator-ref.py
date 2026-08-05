import onnx,numpy as np,struct
from onnx import numpy_helper
g=onnx.load('/tmp/kokoro/model.onnx').graph
I={w.name:numpy_helper.to_array(w).astype(np.float32) for w in g.initializer}
role={}
for n in g.node:
    if n.op_type in ('Conv','Gemm','ConvTranspose') and n.name:
        w=[i for i in n.input if i in I];b=(I[n.input[2]] if len(n.input)>2 and n.input[2] in I else None)
        if w: role[n.name]=(I[w[0]],b)
def R(s):
    for k in role:
        if k.endswith(s):return role[k]
rd=lambda p:np.frombuffer(open(p,'rb').read(),dtype=np.float32)
sty=rd('/tmp/kokoro/hello_style.bin')[:128]
def conv1d(x,W,b,pad,stride=1,dil=1):
    Co,Ci,K=W.shape;xp=np.pad(x,((0,0),(pad,pad)));idx=np.arange(K)*dil;T=(xp.shape[1]-((K-1)*dil+1))//stride+1;o=np.zeros((Co,T),np.float32);Wf=W.reshape(Co,-1)
    for t in range(T):o[:,t]=Wf@xp[:,t*stride+idx].reshape(-1)
    o=o+(b[:,None] if b is not None else 0)
    if stride==1 and o.shape[1]!=x.shape[1]:o=o[:,:x.shape[1]] if o.shape[1]>x.shape[1] else np.pad(o,((0,0),(0,x.shape[1]-o.shape[1])))
    return o
def convT(x,W,b,s,pad):
    Ci,Co,K=W.shape;L=x.shape[1];Lo=(L-1)*s+K;o=np.zeros((Co,Lo),np.float32)
    for t in range(L):
        for k in range(K):o[:,t*s+k]+=W[:,:,k].T@x[:,t]
    return o[:,pad:Lo-pad]+b[:,None]
def leaky(x):return np.where(x>=0,x,0.1*x)
def leaky01(x):return np.where(x>=0,x,0.01*x)  # torch F.leaky_relu default, used before conv_post
def adain(x,fc):
    W,b=R(fc);h=W@sty+(b if b is not None else 0);C=x.shape[0];mu=x.mean(1,keepdims=True);v=x.var(1,keepdims=True);return (1+h[:C,None])*(x-mu)/np.sqrt(v+1e-5)+h[C:,None]
def snake(x,a):a=a.reshape(-1,1);return x+(1.0/(a+1e-9))*np.sin(a*x)**2
def resb(x,pre):
    K=R(f'{pre}/convs1.0/Conv')[0].shape[2]
    for j,d in enumerate((1,3,5)):
        xt=snake(adain(x,f'{pre}/adain1.{j}/fc/Gemm'),I[f'decoder.decoder.generator.{pre}.alpha1.{j}'])
        W,b=R(f'{pre}/convs1.{j}/Conv');xt=conv1d(xt,W,b,(K-1)*d//2,dil=d)
        xt=snake(adain(xt,f'{pre}/adain2.{j}/fc/Gemm'),I[f'decoder.decoder.generator.{pre}.alpha2.{j}'])
        W,b=R(f'{pre}/convs2.{j}/Conv');xt=conv1d(xt,W,b,(K-1)//2)
        x=x+xt
    return x
def group(x,a,b,c):return (resb(x,a)+resb(x,b)+resb(x,c))/3
def merge(ups,noise):  # reflect-pad ups to noise length, add
    if ups.shape[1]<noise.shape[1]:ups=np.pad(ups,((0,0),(noise.shape[1]-ups.shape[1],0)),mode='reflect')
    L=min(ups.shape[1],noise.shape[1]);return ups[:,:L]+noise[:,:L]
spec=rd('/tmp/kokoro/c3.bin').reshape(22,-1)  # exact source spec
x=rd('/tmp/kokoro/real_decode3.bin').reshape(512,178)
x=convT(leaky(x),*R('ups.0/ConvTranspose'),10,5)
x=merge(x,resb(conv1d(spec,*R('noise_convs.0/Conv'),pad=3,stride=6),'noise_res.0'))
x=group(x,'resblocks.0','resblocks.1','resblocks.2')
g2=np.frombuffer(open('/tmp/kokoro/g2_preups1.bin','rb').read(),dtype=np.float32).reshape(256,1780);xl=leaky(x);mm=min(xl.shape[1],1780);print('pre-ups.1',round(float(np.abs(xl[:,:mm]-g2[:,:mm]).max()),4),'len',x.shape[1])
x=convT(leaky(x),*R('ups.1/ConvTranspose'),6,3);u1=np.frombuffer(open('/tmp/kokoro/l1_ups1.bin','rb').read(),dtype=np.float32).reshape(128,10680);mm=min(x.shape[1],10680);print('ups.1',round(float(np.abs(x[:,:mm]-u1[:,:mm]).max()),4),'len',x.shape[1])
nz=resb(conv1d(spec,*R('noise_convs.1/Conv'),pad=0,stride=1),'noise_res.1');nr=np.frombuffer(open('/tmp/kokoro/nr1.bin','rb').read(),dtype=np.float32).reshape(128,-1);mm=min(nz.shape[1],nr.shape[1]);print('L1 noise vs nr1',round(float(np.abs(nz[:,:mm]-nr[:,:mm]).max()),4),'len',nz.shape[1])
x=merge(x,nz)
rb3=np.frombuffer(open('/tmp/kokoro/l1_rb3in.bin','rb').read(),dtype=np.float32).reshape(128,-1);mm=min(x.shape[1],rb3.shape[1]);print('L1 merged vs rb3in',round(float(np.abs(x[:,:mm]-rb3[:,:mm]).max()),4))
x=group(x,'resblocks.3','resblocks.4','resblocks.5')
cp=np.frombuffer(open('/tmp/kokoro/real_convpost.bin','rb').read(),dtype=np.float32).reshape(22,-1)
x=conv1d(leaky01(x),*R('conv_post/Conv'),pad=3);mm=min(x.shape[1],cp.shape[1]);print('conv_post',round(float(np.abs(x[:,:mm]-cp[:,:mm]).max()),4))
mag=np.exp(x[0:11]);p=np.sin(x[11:22]);x=np.concatenate([mag*np.cos(p),mag*np.sin(p)],0)  # STFT recombine [22,T]
ib=I['decoder.decoder.generator.stft.istft.stft.inverse_basis'];ws=I['decoder.decoder.generator.stft.istft.stft.window_sum']
T=x.shape[1];hop=5;nfft=20;Lo=(T-1)*hop+nfft;wav=np.zeros(Lo);wsum=np.zeros(Lo)
for t in range(T):wav[t*hop:t*hop+nfft]+=ib[:,0,:].T@x[:,t];wsum[t*hop:t*hop+nfft]+=ws
wav=(6.0*wav/np.where(wsum>1e-9,wsum,1))[nfft//2:Lo-nfft//2]  # 6.0 = fixed iSTFT norm (=4.0 Mul × window scale)
ref=rd('/tmp/kokoro/real_wav.bin');m=min(len(wav),len(ref))
print("corr",round(float(np.corrcoef(wav[:m],ref[:m])[0,1]),5),"maxΔ",round(float(np.abs(wav[:m]-ref[:m]).max()),5),"rms my",round(float(np.sqrt((wav[:m]**2).mean())),4),"ref",round(float(np.sqrt((ref[:m]**2).mean())),4))
def sv(p,a,sr=24000):a=np.clip(a,-1,1);d=(a*32767).astype('<i2').tobytes();open(p,'wb').write(b'RIFF'+struct.pack('<I',36+len(d))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,1,1,sr,sr*2,2,16)+b'data'+struct.pack('<I',len(d))+d)
sv('/tmp/kokoro/myfinal.wav',wav);print("saved myfinal.wav")
