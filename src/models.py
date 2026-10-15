"""model components for the looped and fixed-depth transformers."""

# adapted from neel nanda's fixed-depth implementation:
# https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py

import torch as t
import torch.nn as nn
import numpy as np
import einops
import torch.nn.functional as F


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
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

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

    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        """build the attention layer.

        args:
            d_model: width of the residual stream.
            num_heads: number of heads.
            d_head: width of each head.
            n_ctx: longest sequence the causal mask covers.
            model: one-element list holding the owning model.
        """
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
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
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
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

