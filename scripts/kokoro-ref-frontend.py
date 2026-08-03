# Extract bert_encoder (768->512 Linear) + capture d_en and final durations refs
# for the wired kokoro.js frontend pipeline.
import onnx, numpy as np, onnxruntime as ort, json, os
from onnx import helper, TensorProto, numpy_helper
m=onnx.load('model.onnx'); init={i.name:numpy_helper.to_array(i) for i in m.graph.initializer}
producer={o:n for n in m.graph.node for o in n.output}
def mm_weight(bias):
    for n in m.graph.node:
        if n.op_type=='Add' and bias in n.input:
            other=[x for x in n.input if x!=bias][0]; mm=producer.get(other)
            if mm and mm.op_type in ('MatMul','Gemm'):
                for x in mm.input:
                    if x in init: return init[x]
    raise KeyError(bias)
os.makedirs('frontend',exist_ok=True)
be_w=mm_weight('encoder.bert_encoder.bias'); be_b=init['encoder.bert_encoder.bias']
np.ascontiguousarray(be_w.astype(np.float32)).tofile('frontend/be_w.bin')
np.ascontiguousarray(be_b.astype(np.float32)).tofile('frontend/be_b.bin')
DEN='/encoder/bert_encoder/Add_output_0'; DUR='/encoder/predictor/ReduceSum_output_0'
for nm in (DEN,DUR): m.graph.output.append(helper.make_tensor_value_info(nm,TensorProto.FLOAT,None))
onnx.save(m,'model_fe.onnx',save_as_external_data=False)
sess=ort.InferenceSession('model_fe.onnx',providers=['CPUExecutionProvider'])
rng=np.random.default_rng(0);seq=48
ids=np.concatenate([[0],rng.integers(1,178,size=seq-2),[0]]).astype(np.int64)[None,:]
den,dur=sess.run([DEN,DUR],{'input_ids':ids,'style':np.zeros((1,256),np.float32),'speed':np.ones((1,),np.float32)})
print('be_w',be_w.shape,'d_en',den.shape,'durations',dur.shape, 'dur[:8]', np.round(dur.ravel()[:8],2).tolist())
np.ascontiguousarray(ids.astype(np.int32)).tofile('frontend/input_ids.bin')
np.ascontiguousarray(den.astype(np.float32)).tofile('frontend/ref_den.bin')
np.ascontiguousarray(dur.astype(np.float32)).tofile('frontend/ref_dur.bin')
json.dump({'seq':seq,'be_in':list(be_w.shape)[0],'be_out':list(be_w.shape)[1],
           'den_shape':list(den.shape),'dur_shape':list(dur.shape)},open('frontend/ref.json','w'),indent=1)
