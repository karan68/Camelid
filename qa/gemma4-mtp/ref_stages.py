import numpy as np, torch, sys
from paths import assistant_dir, workdir
from mtp_inputs import gen, KV_LEN, POSITION_ID
from transformers import Gemma4AssistantForCausalLM
MD = str(assistant_dir())
torch.set_num_threads(4)
def as_bf16(b): return torch.from_numpy(b.astype('<u2',copy=False)).view(torch.bfloat16)
g = gen()
m = Gemma4AssistantForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
cap = {}
def hook(name):
    def f(_mod,_inp,out):
        v = out[0] if isinstance(out,tuple) else out
        cap[name] = v.detach().float().reshape(-1).numpy()
    return f
m.pre_projection.register_forward_hook(hook('pre_projection'))
for i,l in enumerate(m.model.layers):
    p=f'L{i}.'
    l.input_layernorm.register_forward_hook(hook(p+'input_norm'))
    l.self_attn.q_proj.register_forward_hook(hook(p+'q_proj'))
    l.self_attn.q_norm.register_forward_hook(hook(p+'q_norm'))
    l.self_attn.o_proj.register_forward_hook(hook(p+'o_proj'))
    l.post_attention_layernorm.register_forward_hook(hook(p+'post_attn_norm'))
    l.pre_feedforward_layernorm.register_forward_hook(hook(p+'pre_ff_norm'))
    l.mlp.down_proj.register_forward_hook(hook(p+'down_proj'))
    l.post_feedforward_layernorm.register_forward_hook(hook(p+'post_ff_norm'))
    l.register_forward_hook(hook(p+'output'))
m.model.norm.register_forward_hook(hook('final_norm'))
m.post_projection.register_forward_hook(hook('post_projection'))
shared={"sliding_attention":(as_bf16(g["sliding_key_layer28"]),as_bf16(g["sliding_value_layer28"])),
        "full_attention":(as_bf16(g["full_key_layer29"]),as_bf16(g["full_value_layer29"]))}
with torch.inference_mode():
    m(inputs_embeds=torch.cat((as_bf16(g["target_scaled_embedding"]),as_bf16(g["target_final_normalized_hidden"])),dim=-1),
      position_ids=torch.tensor([[POSITION_ID]],dtype=torch.long),
      attention_mask=torch.ones((1,KV_LEN),dtype=torch.long), shared_kv_states=shared, use_cache=False)
np.savez(workdir() / 'stages.npz', **cap)
print(f"captured {len(cap)} stages")
for k in ['pre_projection','L0.input_norm','L0.q_proj','L0.q_norm','L0.o_proj','L0.output','final_norm','post_projection']:
    print(f"  {k:22s} n={cap[k].size:6d} |.|={np.linalg.norm(cap[k]):.4f}")
