# Extract one Kokoro bidirectional LSTM (predictor/lstm: input 640, hidden 256)
# + capture its input X and output Y via ORT, for raw-WebGPU parity.
#   cd /tmp/kokoro && python3 kokoro-ref-lstm.py   # -> lstm/*.bin
import onnx, numpy as np, onnxruntime as ort, json, os
from onnx import helper, TensorProto, numpy_helper
m = onnx.load('model.onnx')
init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
NODE='/encoder/predictor/lstm/LSTM'
node=next(n for n in m.graph.node if n.name==NODE)
X, W, R, B = node.input[0], node.input[1], node.input[2], node.input[3]
Yname=node.output[0]
os.makedirs('lstm', exist_ok=True)
for nm,t in [('W',W),('R',R),('B',B)]:
    np.ascontiguousarray(init[t].astype(np.float32)).tofile(f'lstm/{nm}.bin')
# expose X and Y
for nm in (X, Yname):
    m.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
onnx.save(m,'model_lstm.onnx', save_as_external_data=False)
sess=ort.InferenceSession('model_lstm.onnx', providers=['CPUExecutionProvider'])
rng=np.random.default_rng(1); seq=24
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
outs=sess.run([X,Yname],{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})
Xv,Yv=outs
print('X',Xv.shape,'Y',Yv.shape,'W',init[W].shape,'R',init[R].shape,'B',init[B].shape)
np.ascontiguousarray(Xv.astype(np.float32)).tofile('lstm/X.bin')
np.ascontiguousarray(Yv.astype(np.float32)).tofile('lstm/Y.bin')
json.dump({'seq':int(Xv.shape[0]),'inp':int(Xv.shape[-1]),'hid':256,
           'X_shape':list(Xv.shape),'Y_shape':list(Yv.shape)}, open('lstm/ref.json','w'),indent=1)
print('Y mean/std', float(Yv.mean()), float(Yv.std()))
