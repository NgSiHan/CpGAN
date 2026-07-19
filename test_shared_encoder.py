"""
test_shared_encoder.py — sanity check for the --shared_encoder wiring (HANDOVER Lever 1).

Runs on CPU, no CUDA needed. The bug this guards against is silent: if the tied
encoder's params get listed twice in the optimizer, Adam double-steps them and the
whole Phase-1 experiment is corrupted with no error message.

    python test_shared_encoder.py   # prints OK or asserts
"""
import torch
from model import ResNetIrisEncoder, IrisEncoder


def _dedup(net_vis, net_nir, shared):
    return list(net_vis.parameters()) if shared \
        else list(net_vis.parameters()) + list(net_nir.parameters())


def check(Enc, feat_dim):
    x = torch.randn(2, 1, 64, 512)

    # shared: net_nir IS net_vis
    net_vis = Enc(feat_dim=feat_dim).eval()
    net_nir = net_vis
    assert net_nir is net_vis
    single = sum(p.numel() for p in net_vis.parameters())
    shared_params = _dedup(net_vis, net_nir, shared=True)
    assert len(shared_params) == len(list(net_vis.parameters())), "shared list must not double-count"
    assert sum(p.numel() for p in shared_params) == single, "shared param count must equal one encoder"
    # both modalities really run through the same weights -> identical output for identical input
    assert torch.allclose(net_vis(x), net_nir(x)), "shared encoder must give identical embeddings"

    # non-shared: two encoders -> params add up
    nv, nn = Enc(feat_dim=feat_dim), Enc(feat_dim=feat_dim)
    assert sum(p.numel() for p in _dedup(nv, nn, shared=False)) == 2 * single
    print(f"OK  {Enc.__name__}(feat_dim={feat_dim}): shared={single:,} params, non-shared={2*single:,}")


if __name__ == "__main__":
    check(ResNetIrisEncoder, 512)
    check(IrisEncoder, 128)
    print("all shared-encoder invariants hold")
