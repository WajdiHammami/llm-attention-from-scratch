import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.modules.rope import RotaryEmbedding, rotate_half


def test_rotate_half_correctness():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # real=[1,2], imag=[3,4]
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    out = rotate_half(x)
    assert torch.allclose(out, expected), f"rotate_half bad: {out}"



def test_rope_rotates_correct_pairs():
    rope = RotaryEmbedding(head_dim=4, max_position_embeddings=16)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)

    cos, sin = rope.get_cos_sin(1, x.device, x.dtype, offset=2)
    q_rot, _ = rope.apply_rotary(x, x, cos, sin)
    q_rot = q_rot[0,0,0]   # (4,)

    c0, c1, c2, c3 = cos[0]
    s0, s1, s2, s3 = sin[0]

    # Manual expected formula
    exp0 = 1*c0 - 3*s0  # (real0*cos - imag0*sin)
    exp2 = 1*s2 + 3*c2  # (real0*sin + imag0*cos)

    exp1 = 2*c1 - 4*s1  # (real1*cos - imag1*sin)
    exp3 = 2*s3 + 4*c3  # (real1*sin + imag1*cos)

    assert torch.allclose(q_rot[0], exp0, atol=1e-5)
    assert torch.allclose(q_rot[2], exp2, atol=1e-5)
    assert torch.allclose(q_rot[1], exp1, atol=1e-5)
    assert torch.allclose(q_rot[3], exp3, atol=1e-5)



def test_rope_commutes_with_expansion():
    rope = RotaryEmbedding(head_dim=4)
    B, g, T, d = 1, 1, 6, 4
    h = 8   # expand KV head 1 → 8 heads

    K = torch.randn(B, g, T, d)

    cos, sin = rope.get_cos_sin(T, K.device, K.dtype)

    # A) rotate then expand
    _, K_rot = rope.apply_rotary(K, K, cos, sin)
    A = K_rot.repeat_interleave(h, dim=1)

    # B) expand then rotate
    K_exp = K.repeat_interleave(h, dim=1)
    _, B_out = rope.apply_rotary(K_exp, K_exp, cos, sin)

    assert torch.allclose(A, B_out, atol=1e-5), "RoPE does NOT commute with expansion!"



def test_rope_position_dependence():
    rope = RotaryEmbedding(4)
    x = torch.randn(1,1,1,4)

    cos0, sin0 = rope.get_cos_sin(1, x.device, x.dtype, offset=0)
    q0, _ = rope.apply_rotary(x, x, cos0, sin0)

    cos5, sin5 = rope.get_cos_sin(1, x.device, x.dtype, offset=5)
    q5, _ = rope.apply_rotary(x, x, cos5, sin5)

    assert not torch.allclose(q0, q5), "RoPE failed: pos0 == pos5 !!"
