import numpy as np, json, struct, sys, math
sys.path.insert(0, r'C:\Users\timto\AppData\Local\Temp\claude\C--Users-timto\b5a0fa2e-2b45-41c9-8bca-f2b517e201ad\scratchpad')
from mtp_inputs import gen, bf16_to_f32, KV_LEN, POSITION_ID

MD = r'C:\Users\timto\Projects\Camelid\models\gemma-4-26B-A4B-it-assistant'
EPS = 1e-6

class ST:
    def __init__(self, path):
        with open(path,'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.hdr.pop('__metadata__', None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
    def f32(self, name):
        e = self.hdr[name]; s,t = e['data_offsets']
        bits = self.mm[self.base+s:self.base+t].view(np.uint16)
        return bf16_to_f32(bits).reshape(e['shape']).astype(np.float32)

def rms_norm(x, w, eps=EPS):
    # Gemma convention: scale by (1 + w)
    return (x * (1.0/np.sqrt((x.astype(np.float32)**2).mean(-1, keepdims=True) + eps))) * (1.0 + w)

def gelu_tanh(x):
    return 0.5*x*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))

def rope(q, pos, theta, head_dim, rotary_dim):
    # q: [heads, head_dim]; rotate first `rotary_dim` dims in half-split form
    half = rotary_dim//2
    idx = np.arange(half, dtype=np.float32)
    inv = 1.0/(theta ** (2.0*idx/rotary_dim))
    ang = pos*inv
    cos, sin = np.cos(ang), np.sin(ang)
    out = q.copy()
    a = q[:, :half]; b = q[:, half:rotary_dim]
    out[:, :half]           = a*cos - b*sin
    out[:, half:rotary_dim] = b*cos + a*sin
    return out

def forward(st, g, dbg=False):
    emb  = bf16_to_f32(g['target_scaled_embedding'].reshape(-1))
    hid  = bf16_to_f32(g['target_final_normalized_hidden'].reshape(-1))
    x = np.concatenate([emb, hid])                      # [5632]
    h = st.f32('pre_projection.weight') @ x             # [1024,5632]@[5632] -> [1024]
    kv = {
      'sliding': (bf16_to_f32(g['sliding_key_layer28'][0]), bf16_to_f32(g['sliding_value_layer28'][0])),
      'full':    (bf16_to_f32(g['full_key_layer29'][0]),    bf16_to_f32(g['full_value_layer29'][0])),
    }
    for L in range(4):
        p = f'model.layers.{L}.'
        sliding = L < 3
        kind = 'sliding' if sliding else 'full'
        hd   = 256 if sliding else 512
        theta = 10000.0 if sliding else 1000000.0
        rot  = hd if sliding else int(hd*0.25)
        k, v = kv[kind]
        kvh  = k.shape[0]; heads = 16
        res = h
        xn = rms_norm(h, st.f32(p+'input_layernorm.weight'))
        q = (st.f32(p+'self_attn.q_proj.weight') @ xn).reshape(heads, hd)
        q = rms_norm(q, st.f32(p+'self_attn.q_norm.weight'))
        q = rope(q, POSITION_ID, theta, hd, rot)
        rep = heads//kvh
        scale = 1.0/math.sqrt(hd)
        ctx = np.empty((heads, hd), dtype=np.float32)
        lo = 0
        if sliding:
            lo = max(0, POSITION_ID - 1024 + 1)   # window over [lo, KV_LEN)
        for hh in range(heads):
            kh = hh//rep
            sc = (k[kh, lo:] @ q[hh]) * scale
            sc -= sc.max(); e = np.exp(sc); e /= e.sum()
            ctx[hh] = e @ v[kh, lo:]
        o = st.f32(p+'self_attn.o_proj.weight') @ ctx.reshape(-1)
        o = rms_norm(o, st.f32(p+'post_attention_layernorm.weight'))
        h = res + o
        res = h
        xn = rms_norm(h, st.f32(p+'pre_feedforward_layernorm.weight'))
        gt = gelu_tanh(st.f32(p+'mlp.gate_proj.weight') @ xn)
        up = st.f32(p+'mlp.up_proj.weight') @ xn
        d  = st.f32(p+'mlp.down_proj.weight') @ (gt*up)
        d  = rms_norm(d, st.f32(p+'post_feedforward_layernorm.weight'))
        h = res + d
        if dbg: print(f'  L{L} |h|={np.linalg.norm(h):.4f} scalar={st.f32(p+"layer_scalar")[0]:.6f}')
    h = rms_norm(h, st.f32('model.norm.weight'))
    recur = st.f32('post_projection.weight') @ h
    return h, recur

if __name__ == '__main__':
    st = ST(MD + r'\model.safetensors')
    g = gen()
    h, recur = forward(st, g, dbg=True)
    z = np.load(r'C:\Users\timto\AppData\Local\Temp\claude\C--Users-timto\b5a0fa2e-2b45-41c9-8bca-f2b517e201ad\scratchpad\oracle.npz')
    want_recur = bf16_to_f32(z['recurrent_hidden_bf16_le'])
    print(f"\nrecurrent_hidden: ours |.|={np.linalg.norm(recur):.4f}  oracle |.|={np.linalg.norm(want_recur):.4f}")
    print(f"  cosine = {float(recur@want_recur/(np.linalg.norm(recur)*np.linalg.norm(want_recur))):.6f}")
    print(f"  max abs diff = {np.abs(recur-want_recur).max():.6f}")
    # logits from tied embeddings, chunked
    E = st.hdr['model.embed_tokens.weight']; s,t = E['data_offsets']
    bits = st.mm[st.base+s:st.base+t].view(np.uint16).reshape(262144,1024)
    logits = np.empty(262144, dtype=np.float32)
    for i in range(0, 262144, 16384):
        logits[i:i+16384] = bf16_to_f32(bits[i:i+16384]) @ h
    top = np.argsort(-logits)[:16]
    print(f"\nours   top8: {top[:8].tolist()}")
    print(f"oracle top8: {z['top16_token_ids'][:8].tolist()}")
    print(f"match: {list(top[:16])==list(z['top16_token_ids'])}")
