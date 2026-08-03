# Extract Kokoro's heavy iSTFTNet upsampler (generator/ups.0 ConvTranspose:
# W[512,256,20], stride10, pad5) + capture its input/output for raw-WebGPU parity.
#   cd /tmp/kokoro && python3 kokoro-ref-convt.py   # -> convt/*.bin
import onnx, numpy as np, onnxruntime as ort, json, os
from onnx import helper, TensorProto, numpy_helper
m=onnx.load('model.onnx')
init={i.name:numpy_helper.to_array(i) for i in m.graph.initializer}
NODE='/decoder/decoder/generator/ups.0/ConvTranspose'
node=next(n for n in m.graph.node if n.name==NODE)
Xn,Wn=node.input[0],node.input[1]; Bn=node.input[2] if len(node.input)>2 else None
Yn=node.output[0]
attrs={a.name:(list(a.ints) if a.ints else a.i) for a in node.attribute}
os.makedirs('convt',exist_ok=True)
np.ascontiguousarray(init[Wn].astype(np.float32)).tofile('convt/W.bin')
if Bn: np.ascontiguousarray(init[Bn].astype(np.float32)).tofile('convt/B.bin')
for nm in (Xn,Yn): m.graph.output.append(helper.make_tensor_value_info(nm,TensorProto.FLOAT,None))
onnx.save(m,'model_convt.onnx',save_as_external_data=False)
sess=ort.InferenceSession('model_convt.onnx',providers=['CPUExecutionProvider'])
rng=np.random.default_rng(2); seq=24
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
Xv,Yv=sess.run([Xn,Yn],{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})
print('X',Xv.shape,'Y',Yv.shape,'W',init[Wn].shape,'attrs',attrs)
np.ascontiguousarray(Xv.astype(np.float32)).tofile('convt/X.bin')
np.ascontiguousarray(Yv.astype(np.float32)).tofile('convt/Y.bin')
json.dump({'Cin':int(Xv.shape[1]),'L':int(Xv.shape[2]),'Cout':int(Yv.shape[1]),'Lout':int(Yv.shape[2]),
           'K':int(init[Wn].shape[2]),'stride':attrs['strides'][0],'pad':attrs['pads'][0],
           'groups':attrs.get('group',1),'outputPadding':(attrs.get('output_padding') or [0])[0],
           'hasBias':bool(Bn)}, open('convt/ref.json','w'),indent=1)
