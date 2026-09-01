import numpy as np, hashlib, math
KV_LEN=1031; POSITION_ID=1031; TARGET_HIDDEN=2816
SLIDING_KV_HEADS=8; SLIDING_HEAD_DIM=256; FULL_KV_HEADS=2; FULL_HEAD_DIM=512
INPUT_SPECS={
 "target_scaled_embedding":((1,1,TARGET_HIDDEN),0x13579BDF,120,3),
 "target_final_normalized_hidden":((1,1,TARGET_HIDDEN),0x2468ACE1,125,3),
 "sliding_key_layer28":((1,SLIDING_KV_HEADS,KV_LEN,SLIDING_HEAD_DIM),0x31415926,125,3),
 "sliding_value_layer28":((1,SLIDING_KV_HEADS,KV_LEN,SLIDING_HEAD_DIM),0x27182818,125,3),
 "full_key_layer29":((1,FULL_KV_HEADS,KV_LEN,FULL_HEAD_DIM),0x16180339,125,3),
 "full_value_layer29":((1,FULL_KV_HEADS,KV_LEN,FULL_HEAD_DIM),0x57721566,125,3),
}
def deterministic_bf16_bits(shape,seed,exponent_base,exponent_span):
    count=math.prod(shape)
    index=np.arange(count,dtype=np.uint32)
    state=index+np.uint32(seed)
    state^=state>>np.uint32(16); state*=np.uint32(0x7FEB352D)
    state^=state>>np.uint32(15); state*=np.uint32(0x846CA68B)
    state^=state>>np.uint32(16)
    sign=((state>>np.uint32(31))&np.uint32(1))<<np.uint32(15)
    exponent=(np.uint32(exponent_base)+((state>>np.uint32(24))%np.uint32(exponent_span)))<<np.uint32(7)
    mantissa=(state>>np.uint32(16))&np.uint32(0x7F)
    return (sign|exponent|mantissa).astype(np.uint16,copy=False).reshape(shape)
def bf16_to_f32(bits):
    return (bits.astype(np.uint32)<<16).view(np.float32).reshape(bits.shape)
def gen():
    return {k:deterministic_bf16_bits(*s) for k,s in INPUT_SPECS.items()}
