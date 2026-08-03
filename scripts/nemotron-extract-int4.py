# Extract one Nemotron MatMulNBits (int4 block-quant) + CPU dequant reference.
import onnx, numpy as np, os
from onnx import numpy_helper
m=onnx.load('encoder.onnx'); init={i.name:numpy_helper.to_array(i) for i in m.graph.initializer}
Bq=init['val_121_Q4']            # [N,nblk,16] u8
sc=init['val_121_scales']        # [N*nblk] f32
zp=init['val_121_zero_point']    # [N*ceil(nblk/2)] u8
N,nblk,blob=Bq.shape; K=nblk*32; zpb=(nblk+1)//2
print('N',N,'K',K,'nblk',nblk,'zpb',zpb)
# dequant W[N,K]
def q_at(n,k):
    b=k//32; j=k%32; byte=Bq[n,b,j>>1]; return (byte>>(4*(j&1)))&0xF
def zp_at(n,b):
    byte=zp[n*zpb + (b>>1)]; return (byte>>(4*(b&1)))&0xF
W=np.zeros((N,K),np.float32)
for n in range(N):
    for b in range(nblk):
        s=sc[n*nblk+b]; z=zp_at(n,b)
        for jj in range(32):
            k=b*32+jj; byte=Bq[n,b,jj>>1]; q=(byte>>(4*(jj&1)))&0xF
            W[n,k]=(float(q)-float(z))*s
rng=np.random.default_rng(0); M=64
A=(rng.standard_normal((M,K))*0.1).astype(np.float32)
Y=A@W.T   # [M,N]
os.makedirs('int4',exist_ok=True)
Bq.astype(np.uint8).tofile('int4/Bq.bin'); sc.astype(np.float32).tofile('int4/scales.bin')
zp.astype(np.uint8).tofile('int4/zp.bin'); A.tofile('int4/A.bin'); Y.astype(np.float32).tofile('int4/Y.bin')
import json; json.dump({'M':M,'N':N,'K':K,'nblk':nblk,'zpb':zpb},open('int4/meta.json','w'))
print('W range',float(W.min()),float(W.max()),'Y range',float(Y.min()),float(Y.max()))
