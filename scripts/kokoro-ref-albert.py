# Capture reference ALBERT input/output from the Kokoro ONNX by exposing the
# intermediate tensors as graph outputs and running onnxruntime on a sample
# input_ids. Produces the ground truth that scripts/gpu-albert.mjs checks against.
#   cd /tmp/kokoro && python3 kokoro-ref-albert.py   # -> albert/{input_ids,ref_*}.bin
import onnx, numpy as np, onnxruntime as ort, json
from onnx import helper, TensorProto
m = onnx.load('model.onnx')
ALBERT_IN  = '/encoder/bert/encoder/embedding_hidden_mapping_in/Add_output_0'
ALBERT_OUT = '/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/LayerNormalization_output_0'
for nm in (ALBERT_IN, ALBERT_OUT):
    m.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
onnx.save(m, 'model_dbg.onnx', save_as_external_data=False)

sess = ort.InferenceSession('model_dbg.onnx', providers=['CPUExecutionProvider'])
rng = np.random.default_rng(0)
seq = 24
ids = np.concatenate([[0], rng.integers(1,178,size=seq-2), [0]]).astype(np.int64)[None,:]  # [1,24]
style = np.zeros((1,256), np.float32)
speed = np.ones((1,), np.float32)
outs = sess.run([ALBERT_IN, ALBERT_OUT], {'input_ids': ids, 'style': style, 'speed': speed})
ain, aout = outs
print('albert_in', ain.shape, 'albert_out', aout.shape)
np.ascontiguousarray(ids.astype(np.int32)).tofile('albert/input_ids.bin')
np.ascontiguousarray(ain.astype(np.float32)).tofile('albert/ref_albert_in.bin')
np.ascontiguousarray(aout.astype(np.float32)).tofile('albert/ref_albert_out.bin')
json.dump({'seq':seq,'ids':ids[0].tolist(),
           'albert_in_shape':list(ain.shape),'albert_out_shape':list(aout.shape)},
          open('albert/ref.json','w'), indent=1)
print('saved. out mean/std:', float(aout.mean()), float(aout.std()))
