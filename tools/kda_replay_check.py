#!/usr/bin/env python3
"""GPU regression check for patch 0109, in the patched TensorFold image.

Run with TF_GLM_KDA_REPLAY=1 and TF_GLM_KDA_REPLAY_LEN=2 or 16. Compares
outputs and committed states to eager KDA across flushes, rejected drafts,
wide accepted rounds, and CUDA graph replays, including a 22-head TP=3 rank.
"""
from types import SimpleNamespace as NS

import torch
from tensorfold.families.glm5_next.cuda import forward, kda


def check(heads, graphed):
    rows, layers, cap = 16, 2, kda.REPLAY_LEN + 16
    device = 'cuda'
    width = heads * 384 + 256 + heads
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, device=device, dtype=dtype)
    def random(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, device=device) * .2).to(dtype)
    scratch = kda.KDAScratchSet(layers, rows, heads, device)
    st = NS(rb_h=0, rb_h_dev=zeros(1, dtype=torch.int32), rb_idx=torch.arange(cap, device=device),
            rb_k=zeros(layers, cap, heads, 128), rb_v=zeros(layers, cap, heads, 128, dtype=torch.bfloat16),
            rb_g=torch.ones(layers, cap, heads, 128, device=device), rb_b=zeros(layers, cap, heads),
            scratch=scratch.views, cur=[0] * layers, rec=zeros(2, layers, heads, 128, 128),
            conv=zeros(layers, 3, heads * 384, dtype=torch.bfloat16),
            proj=zeros(layers, rows, width, dtype=torch.bfloat16), pos=0)
    st.set_pos = lambda p: setattr(st, 'pos', p)
    w = NS(cfg=NS(lower=-5., eps=1e-6))
    weights = [NS(kda=NS(a_log=zeros(heads), b_off=heads * 384 + 256,
                        conv=random(heads * 384, 4), dt_bias=zeros(heads * 128),
                        norm=torch.ones(128, device=device, dtype=torch.bfloat16))) for _ in range(layers)]
    b = NS(prefill=False, ka=random(rows, heads * 128), kg=random(rows, heads * 128))
    eager = kda.KDAScratchSet(layers, rows, heads, device)
    rec, conv = st.rec[0].clone(), st.conv.clone()
    kda.reserve(rows, heads, device)
    def run(n):
        for li, layer in enumerate(weights):
            forward._kda_replay(layer, w, st, b, n, li, st.cur[li], st.proj[li, :n])
    graphs = {}
    if graphed:
        for n in (1, 3, 4, 16):
            for parity in (0, 1):
                for flush in ((False, True) if n <= kda.REPLAY_LEN else (True,)):
                    st.cur = [parity] * layers
                    st.rb_h = kda.REPLAY_LEN if flush else 0
                    st.rb_h_dev.fill_(st.rb_h)
                    run(n)
                    torch.cuda.synchronize()
                    st.rb_h_dev.fill_(st.rb_h)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run(n)
                    graphs[n, parity, flush] = g
        st.rec.zero_(); st.rb_h = 0; st.rb_h_dev.zero_(); st.cur = [0] * layers
        forward._replay_neutral(st, 0)
    for n, keep in [(1, 1), (3, 2), (4, 1), (16, 16), (1, 1), (3, 3), (16, 7), (4, 4)] * 2:
        st.proj.copy_(random(layers, rows, width))
        b.ka.copy_(random(rows, heads * 128)); b.kg.copy_(random(rows, heads * 128))
        if graphed:
            graphs[n, st.cur[0], st.rb_h + n > kda.REPLAY_LEN].replay()
        else:
            run(n)
        for li, layer in enumerate(weights):
            k = layer.kda
            next_state = torch.empty_like(rec[li])
            out = kda.chain(st.proj[li, :n], k.b_off, b.ka[:n], b.kg[:n], conv[li], k.conv,
                            rec[li], k.a_log, k.dt_bias, k.norm, w.cfg.eps, w.cfg.lower,
                            n, eager.views[li], next_state)
            assert torch.equal(out, scratch.out[li, :n]), (heads, graphed, n, keep, li, 'outputs')
            kda.replay(rec[li], eager.views[li], keep, next_state)
            rec[li].copy_(next_state)
        forward._shift_conv(conv, st.proj, keep)
        forward.commit(w, st, b, n, keep)
        for li in range(layers):
            materialized = torch.empty_like(rec[li])
            kda._ext().replay(st.rec[st.cur[li], li], st.rb_k[li], st.rb_v[li], st.rb_g[li], st.rb_b[li],
                              cap, materialized)
            assert torch.equal(materialized, rec[li]), (heads, graphed, n, keep, li, 'state')
        assert torch.equal(st.conv, conv)
    print(f'PASS heads={heads} graphed={graphed} L={kda.REPLAY_LEN}: outputs and states bitwise equal', flush=True)


if __name__ == '__main__':
    assert kda.REPLAY_LEN > 0, 'Set TF_GLM_KDA_REPLAY=1'
    torch.manual_seed(19)
    with torch.no_grad():
        for heads in (32, 22):
            for graphed in (False, True):
                check(heads, graphed)
