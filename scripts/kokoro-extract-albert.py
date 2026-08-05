# Extract Kokoro's ALBERT (PL-BERT) weights from the ONNX model into raw f32 bins.
# The nn.Linear weights are anonymous initializers (onnx::MatMul_*), so trace each
# via the named bias it feeds. Run from a dir holding model.onnx:
#   cd /tmp/kokoro && python3 kokoro-extract-albert.py   # -> albert/*.bin + manifest.json
# model.onnx = onnx-community/Kokoro-82M-v1.0-ONNX : onnx/model.onnx
import onnx, numpy as np, json, os
from onnx import numpy_helper
m = onnx.load('model.onnx')
init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
producer = {o: n for n in m.graph.node for o in n.output}

def matmul_weight_for_bias(bias_name):
    # find Add consuming bias_name; its other input is a MatMul output; return that MatMul's initializer input
    for n in m.graph.node:
        if n.op_type == 'Add' and bias_name in n.input:
            other = [x for x in n.input if x != bias_name][0]
            mm = producer.get(other)
            if mm and mm.op_type in ('MatMul','Gemm'):
                for x in mm.input:
                    if x in init: return init[x]
    raise KeyError(bias_name)

# BERT namespace: v1.0/en = 'encoder.bert', v1.1-zh = 'kmodel.bert' (KOKORO_BERT_PREFIX).
B = os.environ.get('KOKORO_BERT_PREFIX', 'encoder.bert')
P = f'{B}.encoder.albert_layer_groups.0.albert_layers.0.'
E = f'{B}.embeddings.'
out = {}
# embeddings
out['word_emb'] = init[E+'word_embeddings.weight']         # [N,128]
out['pos_emb']  = init[E+'position_embeddings.weight']     # [512,128]
out['tok_emb']  = init[E+'token_type_embeddings.weight']   # [2,128]
out['emb_ln_w'] = init[E+'LayerNorm.weight']; out['emb_ln_b'] = init[E+'LayerNorm.bias']
# 128->768 mapping
out['map_w'] = matmul_weight_for_bias(f'{B}.encoder.embedding_hidden_mapping_in.bias')  # [128,768]
out['map_b'] = init[f'{B}.encoder.embedding_hidden_mapping_in.bias']
# shared albert layer
for nm,bias in [('q','attention.query.bias'),('k','attention.key.bias'),('v','attention.value.bias'),
                ('dense','attention.dense.bias'),('ffn','ffn.bias'),('ffn_out','ffn_output.bias')]:
    out[nm+'_w'] = matmul_weight_for_bias(P+bias)
    out[nm+'_b'] = init[P+bias]
out['attn_ln_w']=init[P+'attention.LayerNorm.weight']; out['attn_ln_b']=init[P+'attention.LayerNorm.bias']
out['full_ln_w']=init[P+'full_layer_layer_norm.weight']; out['full_ln_b']=init[P+'full_layer_layer_norm.bias']

os.makedirs('albert', exist_ok=True)
manifest={}
for k,v in out.items():
    v=np.ascontiguousarray(v.astype(np.float32))
    v.tofile(f'albert/{k}.bin'); manifest[k]=list(v.shape)
json.dump(manifest, open('albert/manifest.json','w'), indent=1)
print('shapes:'); [print(f'  {k}: {s}') for k,s in manifest.items()]
