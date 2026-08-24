import numpy as np, torch, sys, json
sys.path.insert(0, r'C:\Users\timto\AppData\Local\Temp\claude\C--Users-timto\b5a0fa2e-2b45-41c9-8bca-f2b517e201ad\scratchpad')
from mtp_inputs import gen, INPUT_SPECS, KV_LEN, POSITION_ID
from transformers import Gemma4AssistantForCausalLM
MD = r'C:\Users\timto\Projects\Camelid\models\gemma-4-26B-A4B-it-assistant'
torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
def as_bf16(bits):
    return torch.from_numpy(bits.astype('<u2', copy=False)).view(torch.bfloat16)
g = gen()
model = Gemma4AssistantForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True)
model.eval()
emb = as_bf16(g["target_scaled_embedding"]); hid = as_bf16(g["target_final_normalized_hidden"])
inputs_embeds = torch.cat((emb, hid), dim=-1)
shared = {"sliding_attention": (as_bf16(g["sliding_key_layer28"]), as_bf16(g["sliding_value_layer28"])),
          "full_attention":    (as_bf16(g["full_key_layer29"]),    as_bf16(g["full_value_layer29"]))}
with torch.inference_mode():
    out = model(inputs_embeds=inputs_embeds,
                position_ids=torch.tensor([[POSITION_ID]], dtype=torch.long),
                attention_mask=torch.ones((1, KV_LEN), dtype=torch.long),
                shared_kv_states=shared, use_cache=False)
logits = out.logits[0,0].float().numpy()
recur  = out.last_hidden_state[0,0].float().numpy()
z = np.load(r'C:\Users\timto\AppData\Local\Temp\claude\C--Users-timto\b5a0fa2e-2b45-41c9-8bca-f2b517e201ad\scratchpad\oracle.npz')
def bf(b): return (b.astype(np.uint32)<<16).view(np.float32)
wr = bf(z['recurrent_hidden_bf16_le']); wl = bf(z['logits_bf16_le'])
cr = float(recur@wr/(np.linalg.norm(recur)*np.linalg.norm(wr)))
print(f"recurrent: cosine={cr:.6f}  |ours|={np.linalg.norm(recur):.2f} |oracle|={np.linalg.norm(wr):.2f}  exact={np.array_equal(recur,wr)}")
top = np.argsort(-logits)[:16]
print(f"top8 ours  : {top[:8].tolist()}")
print(f"top8 oracle: {z['top16_token_ids'][:8].tolist()}")
print(f"top16 match: {list(top)==list(z['top16_token_ids'])}")
np.save(r'C:\Users\timto\AppData\Local\Temp\claude\C--Users-timto\b5a0fa2e-2b45-41c9-8bca-f2b517e201ad\scratchpad\ref_recur.npy', recur)
