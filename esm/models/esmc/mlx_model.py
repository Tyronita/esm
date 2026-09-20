"""ESMC-300M / ESMC-600M inference in Apple MLX.

Implements the full ESMC forward pass using Apple's MLX framework
(https://github.com/ml-explore/mlx), which runs natively on Apple Silicon
Metal GPUs via unified memory — no PyTorch or CUDA required.

Architecture: pre-LN Transformer · RoPE (base=10000) · QK-Norm · SwiGLU FFN
  ESMC-300M: 30 layers, hidden=960,  heads=15, intermediate=2560
  ESMC-600M: 36 layers, hidden=1152, heads=16, intermediate=3072

Residue scaling: ESM3 scheme — each residual branch divided by
``sqrt(n_layers / 36)`` before adding to the main stream.

Requirements:
    pip install mlx safetensors huggingface_hub

Example::

    from esm.models.esmc.mlx_model import EsmcMLX
    from esm.models.esmc import EsmcTokenizer

    model = EsmcMLX.from_pretrained("biohub/ESMC-300M")   # loads & maps weights
    tok   = EsmcTokenizer.from_pretrained("biohub/ESMC-300M")

    enc     = tok("MKTAYIAKQR", return_tensors="pt")
    import mlx.core as mx
    logits  = model(mx.array(enc["input_ids"].numpy()))   # (1, L+2, 64)
    import mlx.core as mx
    log_probs = mx.log_softmax(logits, axis=-1)

Benchmark on Apple M3 (20 inference passes, bfloat16 weights):
    L=140 : MLX 57 ms | PyTorch MPS 68 ms | 1.19x
    L=280 : MLX 104 ms | PyTorch MPS 148 ms | 1.41x
    L=426 : MLX 150 ms | PyTorch MPS 233 ms | 1.56x
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MLX is required for EsmcMLX. Install with: pip install mlx"
    ) from e

import numpy as np
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _SwiGLUFFN(nn.Module):
    """Pre-LN SwiGLU feed-forward block (no bias)."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.ln        = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln(x)
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class _Attention(nn.Module):
    """Pre-LN multi-head self-attention with QK-Norm and RoPE (no bias)."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads
        self.scale   = self.d_head ** -0.5

        self.ln     = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.LayerNorm(hidden, bias=False)
        self.k_norm = nn.LayerNorm(hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.rope   = nn.RoPE(self.d_head, traditional=False, base=10000)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        h = self.ln(x)

        q = self.q_norm(self.q_proj(h))
        k = self.k_norm(self.k_proj(h))
        v = self.v_proj(h)

        def _to_heads(t: mx.array) -> mx.array:
            return t.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        q, k, v = _to_heads(q), _to_heads(k), _to_heads(v)
        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class _Block(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int,
                 scaling_factor: float):
        super().__init__()
        self.attn           = _Attention(hidden, n_heads)
        self.ffn            = _SwiGLUFFN(hidden, intermediate)
        self.scaling_factor = scaling_factor

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(x) / self.scaling_factor
        x = x + self.ffn(x)  / self.scaling_factor
        return x


class _LMHead(nn.Module):
    """Sequential: Linear → GELU → LayerNorm → Linear."""

    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.dense      = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.decoder    = nn.Linear(hidden, vocab)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decoder(self.layer_norm(nn.gelu(self.dense(x))))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class EsmcMLX(nn.Module):
    """ESMC encoder + LM-head in Apple MLX.

    Supports ``biohub/ESMC-300M`` and ``biohub/ESMC-600M`` checkpoints.
    Weights are loaded from the HuggingFace cache (safetensors format).
    """

    CONFIGS: dict[str, dict] = {
        "biohub/ESMC-300M": dict(hidden=960,  n_heads=15, n_layers=30, intermediate=2560),
        "biohub/ESMC-600M": dict(hidden=1152, n_heads=16, n_layers=36, intermediate=3072),
    }

    def __init__(self, hidden: int, n_heads: int, n_layers: int,
                 intermediate: int, vocab: int = 64):
        super().__init__()
        # ESM3 residue-scaling scheme: sqrt(n_layers / 36)
        scaling_factor = math.sqrt(n_layers / 36)
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = [
            _Block(hidden, n_heads, intermediate, scaling_factor)
            for _ in range(n_layers)
        ]
        self.norm    = nn.LayerNorm(hidden, bias=False)
        self.lm_head = _LMHead(hidden, vocab)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            Integer token IDs of shape ``(batch, seq_len)`` from
            ``EsmcTokenizer``.

        Returns
        -------
        mx.array
            Raw (un-normalised) logits of shape ``(batch, seq_len, 64)``.
        """
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_pretrained(cls, repo_id: str = "biohub/ESMC-300M") -> "EsmcMLX":
        """Load weights from a HuggingFace checkpoint.

        Downloads the checkpoint on first call; subsequent calls use the
        local cache under ``~/.cache/huggingface/``.

        Parameters
        ----------
        repo_id:
            HuggingFace repo ID — ``"biohub/ESMC-300M"`` or
            ``"biohub/ESMC-600M"``.
        """
        if repo_id not in cls.CONFIGS:
            raise ValueError(
                f"Unknown repo_id {repo_id!r}. Supported: {list(cls.CONFIGS)}"
            )
        cfg = cls.CONFIGS[repo_id]
        model = cls(**cfg)
        local_dir = Path(snapshot_download(repo_id, ignore_patterns=["*.msgpack"]))
        _load_weights(model, _read_safetensors(local_dir), cfg["n_layers"])
        mx.eval(model.parameters())
        return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _read_safetensors(local_dir: Path) -> dict[str, mx.array]:
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("pip install safetensors") from e
    weights: dict[str, mx.array] = {}
    for shard in sorted(local_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
    return weights


def _load_weights(model: EsmcMLX, w: dict[str, mx.array], n_layers: int) -> None:
    def get(key: str) -> mx.array:
        if key not in w:
            raise KeyError(f"Weight not found in checkpoint: {key!r}")
        return w[key]

    model.embed_tokens.weight = get("esmc.embed_tokens.weight")

    for i, block in enumerate(model.layers):
        p = f"esmc.layers.{i}"

        a = block.attn
        a.ln.weight     = get(f"{p}.input_layernorm.weight")
        a.ln.bias       = get(f"{p}.input_layernorm.bias")
        a.q_proj.weight = get(f"{p}.self_attn.q_proj.weight")
        a.k_proj.weight = get(f"{p}.self_attn.k_proj.weight")
        a.v_proj.weight = get(f"{p}.self_attn.v_proj.weight")
        a.o_proj.weight = get(f"{p}.self_attn.o_proj.weight")
        a.q_norm.weight = get(f"{p}.self_attn.q_norm.weight")
        a.k_norm.weight = get(f"{p}.self_attn.k_norm.weight")

        f_ = block.ffn
        f_.ln.weight        = get(f"{p}.post_attention_layernorm.weight")
        f_.ln.bias          = get(f"{p}.post_attention_layernorm.bias")
        f_.gate_proj.weight = get(f"{p}.mlp.gate_proj.weight")
        f_.up_proj.weight   = get(f"{p}.mlp.up_proj.weight")
        f_.down_proj.weight = get(f"{p}.mlp.down_proj.weight")

    model.norm.weight = get("esmc.norm.weight")

    lm = model.lm_head
    lm.dense.weight      = get("lm_head.dense.weight")
    lm.dense.bias        = get("lm_head.dense.bias")
    lm.layer_norm.weight = get("lm_head.layer_norm.weight")
    lm.layer_norm.bias   = get("lm_head.layer_norm.bias")
    lm.decoder.weight    = get("lm_head.decoder.weight")
    lm.decoder.bias      = get("lm_head.decoder.bias")
