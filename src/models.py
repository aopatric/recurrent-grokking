"""model components for the looped and fixed-depth transformers."""

# adapted from neel nanda's fixed-depth implementation:
# https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py

import torch as t
import torch.nn as nn
import numpy as np
import einops
import torch.nn.functional as F
from dataclasses import dataclass, fields


class HookPoint(nn.Module):
    """identity module that exposes an activation to hooks, tagged with its name and loop iteration."""

    def __init__(self):
        """start with no hooks and no iteration tag."""
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.iteration = None

    def give_name(self, name):
        """set the name passed to hooks.

        args:
            name: hook point name, usually its path in the model.
        """
        self.name = name

    def set_iteration(self, iteration):
        """set the loop iteration passed to hooks on the next forward call.

        args:
            iteration: loop iteration index, or None outside the recurrent core.
        """
        self.iteration = iteration

    def add_hook(self, hook, dir='fwd'):
        """attach a hook, called as hook(tensor, name=..., iteration=...).

        args:
            hook: function of the activation ("fwd") or its gradient ("bwd"); a non-None return replaces it.
            dir: "fwd" or "bwd".
        raises:
            ValueError: if dir is not "fwd" or "bwd".
        """
        if dir == 'fwd':
            def full_hook(module, module_input, module_output):
                """forward the module output to hook with its tags."""
                return hook(module_output, name=self.name, iteration=self.iteration)
            self.fwd_hooks.append(self.register_forward_hook(full_hook))
        elif dir == 'bwd':
            # registered as tensor hooks in forward, see there
            self.bwd_hooks.append(hook)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        """detach hooks.

        args:
            dir: "fwd", "bwd" or "both".
        raises:
            ValueError: if dir is not one of these.
        """
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
        if dir in ('fwd', 'both'):
            for handle in self.fwd_hooks:
                handle.remove()
            self.fwd_hooks = []
        if dir in ('bwd', 'both'):
            self.bwd_hooks = []

    def forward(self, x):
        """pass x through unchanged, attaching any gradient hooks to it.

        args:
            x: activation tensor.
        returns:
            x.
        """
        # a module backward hook fires after the whole loop has run, when self.iteration is stale,
        # so gradient hooks go on this call's tensor with the iteration captured now
        if self.bwd_hooks and x.requires_grad:
            name, it = self.name, self.iteration
            for hook in self.bwd_hooks:
                x.register_hook(lambda grad, hook=hook: hook(grad, name=name, iteration=it))
        return x


def set_iteration(module, iteration):
    """tag every hook point inside a module with the current loop iteration.

    args:
        module: module to walk, usually the recurrent core.
        iteration: loop iteration index, or None to clear.
    """
    for m in module.modules():
        if isinstance(m, HookPoint):
            m.set_iteration(iteration)


class UntiedEmbedding(nn.Module):
    """token embedding with its own matrix."""

    def __init__(self, d_vocab, d_model):
        """build the embedding.

        args:
            d_vocab: number of input tokens.
            d_model: width of the residual stream.
        """
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def embed(self, x):
        """embed token ids.

        args:
            x: token ids of shape (batch, pos).
        returns:
            embeddings of shape (batch, pos, d_model).
        """
        return t.einsum('dbp -> bpd', self.W_E[:, x])


class UntiedUnembedding(nn.Module):
    """output projection with its own matrix."""

    def __init__(self, d_vocab, d_model):
        """build the unembedding.

        args:
            d_vocab: number of output classes.
            d_model: width of the residual stream.
        """
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def unembed(self, x):
        """map the residual stream to logits.

        args:
            x: residual stream of shape (..., d_model).
        returns:
            logits of shape (..., d_vocab).
        """
        return x @ self.W_U


class TiedEmbedding(nn.Module):
    """token embedding whose transpose is also the output projection."""

    def __init__(self, d_vocab, d_model):
        """build the shared matrix.

        args:
            d_vocab: number of tokens, shared by input and output.
            d_model: width of the residual stream.
        """
        super().__init__()
        self.W_embed = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def embed(self, x):
        """embed token ids.

        args:
            x: token ids of shape (batch, pos).
        returns:
            embeddings of shape (batch, pos, d_model).
        """
        return t.einsum('dbp -> bpd', self.W_embed[:, x])

    def unembed(self, x):
        """map the residual stream to logits with the shared matrix.

        args:
            x: residual stream of shape (..., d_model).
        returns:
            logits of shape (..., d_vocab).
        """
        return x @ self.W_embed.T


class PositionalEmbedding(nn.Module):
    """learned absolute positional embedding."""

    def __init__(self, max_ctx, d_model):
        """build the position table.

        args:
            max_ctx: longest sequence the table covers.
            d_model: width of the residual stream.
        """
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))

    def embed(self, x):
        """add positional embeddings.

        args:
            x: token embeddings of shape (batch, pos, d_model).
        returns:
            x plus the embedding of each position.
        """
        return x + self.W_pos[:x.shape[-2]]


class LayerNorm(nn.Module):
    """layer norm that the owning model can switch off with its use_ln flag."""

    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        """build the norm.

        args:
            d_model: width of the residual stream.
            epsilon: added to the standard deviation for stability.
            model: one-element list holding the owning model, read for use_ln.
        """
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        """normalize over the model dimension if the model uses layer norm.

        args:
            x: tensor of shape (..., d_model).
        returns:
            normalized x, or x unchanged when use_ln is off.
        """
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


class RMSNorm(nn.Module):
    """root-mean-square norm with a learned gain, for the huginn sandwich blocks."""

    def __init__(self, d_model, epsilon=1e-6):
        """build the norm.

        args:
            d_model: width of the residual stream.
            epsilon: added to the mean square for stability.
        """
        super().__init__()
        self.w_rms = nn.Parameter(t.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        """normalize over the model dimension.

        args:
            x: tensor of shape (..., d_model).
        returns:
            tensor of the same shape with unit rms per position, times the gain.
        """
        return x * t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon) * self.w_rms


class Attention(nn.Module):
    """causal multi-head attention with hook points on its intermediate activations."""

    def __init__(self, d_model, num_heads, d_head, model):
        """build the attention layer.

        args:
            d_model: width of the residual stream.
            num_heads: number of heads.
            d_head: width of each head.
            model: one-element list holding the owning model.
        """
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        """attend over earlier positions.

        args:
            x: residual stream of shape (batch, pos, d_model).
        returns:
            attention output of shape (batch, pos, d_model).
        """
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        # mask built per batch so prompts of any length work without positional embeddings
        n = x.shape[-2]
        mask = t.tril(t.ones((n, n), device=x.device, dtype=x.dtype))
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - mask)
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class MLP(nn.Module):
    """two-layer relu mlp with hook points before and after the nonlinearity."""

    def __init__(self, d_model, d_mlp, model):
        """build the mlp.

        args:
            d_model: width of the residual stream.
            d_mlp: number of hidden neurons.
            model: one-element list holding the owning model.
        """
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        """apply the mlp at every position.

        args:
            x: residual stream of shape (batch, pos, d_model).
        returns:
            mlp output of shape (batch, pos, d_model).
        """
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        x = F.relu(x)
        x = self.hook_post(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x



class Block(nn.Module):
    """one transformer layer: attention then mlp, with no norm (nanda) or a rms sandwich (huginn)."""

    def __init__(self, d_model, num_heads, d_head, d_mlp, norm, model):
        """build the layer.

        args:
            d_model: width of the residual stream.
            num_heads: number of attention heads.
            d_head: width of each head.
            d_mlp: number of mlp neurons.
            norm: "none" or "rms_sandwich".
            model: one-element list holding the owning model.
        raises:
            ValueError: if norm is unknown.
        """
        super().__init__()
        if norm not in ("none", "rms_sandwich"):
            raise ValueError(f"unknown norm {norm!r}")
        self.norm = norm
        self.attn = Attention(d_model, num_heads, d_head, model=model)
        self.mlp = MLP(d_model, d_mlp, model=model)
        if norm == "rms_sandwich":
            # n3 is redundant after n2, but huginn keeps it and so do we
            self.n1, self.n2, self.n3, self.n4 = (RMSNorm(d_model) for _ in range(4))
        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        """apply the layer.

        args:
            x: residual stream of shape (batch, pos, d_model).
        returns:
            residual stream of the same shape.
        """
        x = self.hook_resid_pre(x)
        if self.norm == "none":
            x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(x)))
            return self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        x = self.hook_resid_mid(self.n2(x + self.hook_attn_out(self.attn(self.n1(x)))))
        return self.hook_resid_post(self.n4(x + self.hook_mlp_out(self.mlp(self.n3(x)))))


@dataclass(frozen=True)
class ModelConfig:
    """settings that pick a model from the ladder; widths default to nanda's."""

    p: int
    pos: str = "none"               # none | learned
    n_ctx: int = 0                  # position table size, only used with pos = learned
    norm: str = "none"              # none | rms_sandwich
    n_prelude: int = 0
    k: int = 1                      # layers in the core block
    core_shared: bool = True
    n_core_copies: int = 1          # distinct core blocks; > 1 only when core_shared is false
    injection: str = "none"         # none | concat | add
    state_init: str = "embed"       # embed | random
    n_coda: int = 0
    state_norm: bool = False        # rms norm on the state after every pass; always on with rms_sandwich
    d_model: int = 128
    num_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512

    @classmethod
    def from_dict(cls, d):
        """build a config from a mapping, rejecting unknown keys.

        args:
            d: mapping of field names to values.
        returns:
            ModelConfig.
        raises:
            ValueError: if a key is unknown or a value is invalid.
        """
        known = {f.name for f in fields(cls)}
        if unknown := set(d) - known:
            raise ValueError(f"unknown model keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self):
        """check that the settings are consistent.

        raises:
            ValueError: listing every violated rule.
        """
        errs = []
        for name, allowed in [("pos", ("none", "learned")), ("norm", ("none", "rms_sandwich")),
                              ("injection", ("none", "concat", "add")), ("state_init", ("embed", "random"))]:
            if getattr(self, name) not in allowed:
                errs.append(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.pos == "learned" and self.n_ctx < 1:
            errs.append("pos = learned needs n_ctx >= 1")
        if self.k < 1 or self.n_prelude < 0 or self.n_coda < 0:
            errs.append("k must be >= 1 and n_prelude, n_coda >= 0")
        if self.core_shared and self.n_core_copies != 1:
            errs.append("a shared core has exactly one copy")
        if not self.core_shared and self.n_core_copies < 1:
            errs.append("an unshared core needs n_core_copies >= 1")
        if self.state_init == "random" and self.injection == "none":
            # a random start with no injection never sees the input
            errs.append("state_init = random needs an injection")
        if errs:
            raise ValueError("; ".join(errs))


class LoopedTransformer(nn.Module):
    """embedding, optional prelude, a looped core block, optional coda, unembedding at the last position."""

    def __init__(self, cfg: ModelConfig):
        """build the model.

        args:
            cfg: model settings.
        """
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.use_ln = False
        d = cfg.d_model
        me = [self]

        def blocks(n):
            """make n fresh layers."""
            return nn.ModuleList(Block(d, cfg.num_heads, cfg.d_head, cfg.d_mlp, cfg.norm, me) for _ in range(n))

        self.embed = UntiedEmbedding(d_vocab=cfg.p + 1, d_model=d)
        self.pos_embed = PositionalEmbedding(cfg.n_ctx, d) if cfg.pos == "learned" else None
        self.prelude = blocks(cfg.n_prelude)
        # core[i] is the block used on pass i (always core[0] when shared)
        self.core = nn.ModuleList(blocks(cfg.k) for _ in range(cfg.n_core_copies))
        if cfg.injection == "concat":
            self.W_A = nn.Parameter(t.randn(d, 2 * d) / np.sqrt(2 * d))
        self.core_norm = RMSNorm(d) if cfg.norm == "rms_sandwich" or cfg.state_norm else None
        self.coda = blocks(cfg.n_coda)
        self.coda_norm = RMSNorm(d) if cfg.norm == "rms_sandwich" else None
        self.unembed = UntiedUnembedding(d_vocab=cfg.p, d_model=d)

        self.hook_embed = HookPoint()
        self.hook_inject = HookPoint()
        self.hook_state = HookPoint()
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    # running

    def initial_state(self, e, generator=None):
        """make the starting state of the loop.

        args:
            e: embedded input of shape (batch, pos, d_model).
            generator: torch generator for the random start, or None.
        returns:
            the starting state, same shape as e.
        """
        if self.cfg.state_init == "embed":
            return e
        # truncated normal at 3 sigma, scaled like nanda's embedding init
        std = 1 / np.sqrt(self.cfg.d_model)
        s = t.randn(e.shape, generator=generator, device=e.device, dtype=e.dtype)
        while (bad := s.abs() > 3).any():
            s[bad] = t.randn(int(bad.sum()), generator=generator, device=e.device, dtype=e.dtype)
        return s * std

    def inject(self, s, e):
        """combine the carried state with the embedded input.

        args:
            s: state of shape (batch, pos, d_model).
            e: embedded input of the same shape.
        returns:
            input to the core block.
        """
        if self.cfg.injection == "concat":
            return t.einsum('dc,bpc->bpd', self.W_A, t.cat([s, e], dim=-1))
        if self.cfg.injection == "add":
            return s + e
        return s

    def readout(self, s, iteration):
        """run the coda and unembed the last position.

        args:
            s: state of shape (batch, pos, d_model).
            iteration: loop iteration the state came from, used to tag coda hooks.
        returns:
            logits of shape (batch, p).
        """
        set_iteration(self.coda, iteration)
        for blk in self.coda:
            s = blk(s)
        set_iteration(self.coda, None)
        if self.coda_norm is not None:
            s = self.coda_norm(s)
        # "=" is always the last position: batches are grouped by hop, with no padding
        return self.unembed.unembed(s[:, -1])

    def forward(self, tokens, n_iter, return_all=False, grad_iters=None, generator=None):
        """run the model for n_iter passes of the core block.

        args:
            tokens: token ids of shape (batch, pos), ending with "=".
            n_iter: number of core passes, >= 1.
            return_all: if True, read out after every pass.
            grad_iters: backprop through only the last this-many passes; None for all.
            generator: torch generator for the random start, or None.
        returns:
            logits (batch, p), or (batch, n_iter, p) when return_all.
        raises:
            ValueError: if n_iter is out of range for this model.
        """
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if not self.cfg.core_shared and n_iter > self.cfg.n_core_copies:
            raise ValueError(f"unshared core has {self.cfg.n_core_copies} copies, asked for {n_iter} passes")
        e = self.embed.embed(tokens)
        if self.pos_embed is not None:
            e = self.pos_embed.embed(e)
        e = self.hook_embed(e)
        for blk in self.prelude:
            e = blk(e)

        s = self.initial_state(e, generator)
        first_grad = 0 if grad_iters is None else n_iter - grad_iters
        readouts = []
        for i in range(n_iter):
            # passes before the truncation window build no graph, so they cost no activation memory
            with t.set_grad_enabled(t.is_grad_enabled() and i >= first_grad):
                set_iteration(self.core, i)
                self.hook_inject.set_iteration(i)
                self.hook_state.set_iteration(i)
                u = self.hook_inject(self.inject(s, e))
                for blk in self.core[i if not self.cfg.core_shared else 0]:
                    u = blk(u)
                if self.core_norm is not None:
                    u = self.core_norm(u)
                s = self.hook_state(u)
                if return_all:
                    readouts.append(self.readout(s, i))
        set_iteration(self.core, None)
        self.hook_inject.set_iteration(None)
        self.hook_state.set_iteration(None)
        if return_all:
            return t.stack(readouts, dim=1)
        return self.readout(s, n_iter - 1)

    # hooks

    def hook_points(self):
        """list every hook point in the model.

        returns:
            list of HookPoint modules.
        """
        return [m for m in self.modules() if isinstance(m, HookPoint)]

    def remove_all_hooks(self):
        """detach every forward and backward hook."""
        for hp in self.hook_points():
            hp.remove_hooks('both')

    def cache_all(self, cache, incl_bwd=False):
        """store every hooked activation (and optionally gradient) in cache.

        args:
            cache: dict filled with (name, iteration) -> tensor; gradients use name + "_grad".
            incl_bwd: also cache gradients.
        """
        def save_hook(tensor, name, iteration):
            """store a detached activation."""
            cache[(name, iteration)] = tensor.detach()

        def save_hook_back(tensor, name, iteration):
            """store a detached gradient."""
            cache[(name + "_grad", iteration)] = tensor.detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')
