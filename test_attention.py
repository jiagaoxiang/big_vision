import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from flax.linen.linear import default_kernel_init, DenseGeneral
from flax.linen import MultiHeadDotProductAttention as MultiHeadDotProductAttentionFlax
import functools
from typing import Optional, Callable, Any
from jax import lax
import pytest

# TransformerEngine version (provided)
TE_AVAILABLE=True
from transformer_engine.jax.flax.transformer import DotProductAttention

class MultiHeadDotProductAttentionTE(nn.Module):
  num_heads: int
  dtype: Optional[any] = None
  kernel_init: any = nn.initializers.xavier_uniform()
  deterministic: Optional[bool] = None
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=None):
    if deterministic is None:
      deterministic = self.deterministic
      if deterministic is None:
        deterministic = False

    dim = inputs_q.shape[-1]
    head_dim = dim // self.num_heads

    q = nn.DenseGeneral((self.num_heads, head_dim), kernel_init=self.kernel_init, dtype=self.dtype, name='query')(inputs_q)
    k = nn.DenseGeneral((self.num_heads, head_dim), kernel_init=self.kernel_init, dtype=self.dtype, name='key')(inputs_kv)
    v = nn.DenseGeneral((self.num_heads, head_dim), kernel_init=self.kernel_init, dtype=self.dtype, name='value')(inputs_kv)

    # q = q.reshape(q.shape[:-1] + (self.num_heads, head_dim))
    # k = k.reshape(k.shape[:-1] + (self.num_heads, head_dim))
    # v = v.reshape(v.shape[:-1] + (self.num_heads, head_dim))
    te_mask_type = 'padding' if mask is not None else 'no_mask'
    attn = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=self.num_heads, num_gqa_groups=self.num_heads,
        attention_dropout=self.dropout_rate,
        dtype=self.dtype,
        qkv_layout='bshd_bshd_bshd',
        attn_mask_type=te_mask_type,
        transpose_batch_sequence=False,
    )(q, k, v, mask=(~mask if mask is not None else None), deterministic=deterministic)

    # attn = attn.reshape(attn.shape[:-2] + (dim,))
    out = nn.DenseGeneral(dim, axis=(-2, -1), kernel_init=self.kernel_init, dtype=self.dtype, name='out')(attn)
    return out


# ============================================================================
# Detailed Comparison Functions
# ============================================================================

def compare_qkv_projections(params_te, params_flax, inputs_q, inputs_kv):
    """Compare Q, K, V projections before attention."""
    print("\n" + "="*80)
    print("COMPARING Q/K/V PROJECTIONS")
    print("="*80)
    
    dim = inputs_q.shape[-1]
    num_heads = params_te['query']['kernel'].shape[1]
    head_dim = dim // num_heads
    
    # Manual projection for TE
    q_te = jnp.einsum('...d,dhf->...hf', inputs_q, params_te['query']['kernel'])
    k_te = jnp.einsum('...d,dhf->...hf', inputs_kv, params_te['key']['kernel'])
    v_te = jnp.einsum('...d,dhf->...hf', inputs_kv, params_te['value']['kernel'])
    
    # Manual projection for Flax
    q_flax = jnp.einsum('...d,dhf->...hf', inputs_q, params_flax['query']['kernel'])
    k_flax = jnp.einsum('...d,dhf->...hf', inputs_kv, params_flax['key']['kernel'])
    v_flax = jnp.einsum('...d,dhf->...hf', inputs_kv, params_flax['value']['kernel'])
    
    print(f"Q shape: {q_te.shape}")
    print(f"K shape: {k_te.shape}")
    print(f"V shape: {v_te.shape}")
    print(f"\nQ max diff: {jnp.max(jnp.abs(q_te - q_flax)):.6e}")
    print(f"K max diff: {jnp.max(jnp.abs(k_te - k_flax)):.6e}")
    print(f"V max diff: {jnp.max(jnp.abs(v_te - v_flax)):.6e}")
    print(f"\nQ, K, V projections are {'IDENTICAL' if jnp.allclose(q_te, q_flax) and jnp.allclose(k_te, k_flax) and jnp.allclose(v_te, v_flax) else 'DIFFERENT'}")
    
    return q_te, k_te, v_te, q_flax, k_flax, v_flax


def compare_attention_scores(q, k):
    """Manually compute attention scores."""
    print("\n" + "="*80)
    print("COMPUTING ATTENTION SCORES")
    print("="*80)
    
    # q, k shape: [batch, seq, num_heads, head_dim]
    # Compute scores: [batch, num_heads, seq_q, seq_k]
    scores = jnp.einsum('...qhd,...khd->...hqk', q, k)
    head_dim = q.shape[-1]
    scores = scores / jnp.sqrt(head_dim)
    
    print(f"Attention scores shape: {scores.shape}")
    print(f"Attention scores mean: {jnp.mean(scores):.6f}")
    print(f"Attention scores std: {jnp.std(scores):.6f}")
    print(f"Attention scores min/max: {jnp.min(scores):.6f} / {jnp.max(scores):.6f}")
    
    return scores


def compare_attention_weights(scores, mask=None):
    """Apply softmax to get attention weights."""
    print("\n" + "="*80)
    print("COMPUTING ATTENTION WEIGHTS")
    print("="*80)
    
    if mask is not None:
        print(f"Applying mask with shape: {mask.shape}")
        # Flax uses False for masked positions
        big_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask, scores, big_neg)
    
    weights = jax.nn.softmax(scores, axis=-1)
    
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum (should be ~1): {jnp.mean(jnp.sum(weights, axis=-1)):.6f}")
    print(f"Attention weights mean: {jnp.mean(weights):.6f}")
    print(f"Attention weights std: {jnp.std(weights):.6f}")
    
    return weights


def compare_attention_output(weights, v):
    """Apply attention weights to values."""
    print("\n" + "="*80)
    print("COMPUTING ATTENTION OUTPUT")
    print("="*80)
    
    # weights: [batch, num_heads, seq_q, seq_k]
    # v: [batch, seq_k, num_heads, head_dim]
    # output: [batch, seq_q, num_heads, head_dim]
    output = jnp.einsum('...hqk,...khd->...qhd', weights, v)
    
    print(f"Attention output shape: {output.shape}")
    print(f"Attention output mean: {jnp.mean(output):.6f}")
    print(f"Attention output std: {jnp.std(output):.6f}")
    
    return output


def detailed_comparison():
    """Run detailed step-by-step comparison."""
    print("\n" + "="*80)
    print("DETAILED ATTENTION COMPARISON: TE vs FLAX")
    print("="*80)
    
    if not TE_AVAILABLE:
        print("TransformerEngine not available. Cannot run comparison.")
        return
    
    # Setup
    batch_size, seq_len, dim = 2, 8, 64
    num_heads = 4
    seed = 42
    
    rng = jax.random.PRNGKey(seed)
    rng_input, rng_init = jax.random.split(rng)
    
    inputs_q = jax.random.normal(rng_input, (batch_size, seq_len, dim))
    inputs_kv = inputs_q  # Self-attention
    
    print(f"\nInput shape: {inputs_q.shape}")
    print(f"Num heads: {num_heads}")
    print(f"Head dim: {dim // num_heads}")
    
    # Initialize both models
    kernel_init = nn.initializers.xavier_uniform()
    
    model_te = MultiHeadDotProductAttentionTE(
        num_heads=num_heads,
        kernel_init=kernel_init,
        deterministic=True,
        dropout_rate=0.0
    )
    
    model_flax = MultiHeadDotProductAttentionFlax(
        num_heads=num_heads,
        kernel_init=kernel_init,
        deterministic=True,
        dropout_rate=0.0
    )
    
    # Use same random seed for initialization
    vars_te = model_te.init(rng_init, inputs_q, inputs_kv)
    vars_flax = model_flax.init(rng_init, inputs_q, inputs_kv)
    
    # Copy Flax parameters to TE to ensure identical weights
    vars_te_modified = {
        'params': {
            'query': vars_flax['params']['query'],
            'key': vars_flax['params']['key'],
            'value': vars_flax['params']['value'],
            'out': vars_flax['params']['out']
        }
    }
    
    print("\n" + "="*80)
    print("PARAMETER COMPARISON")
    print("="*80)
    print("Using identical parameters for both models (copied from Flax to TE)")
    
    # Compare Q/K/V projections
    q_te, k_te, v_te, q_flax, k_flax, v_flax = compare_qkv_projections(
        vars_te_modified['params'], 
        vars_flax['params'],
        inputs_q,
        inputs_kv
    )
    
    # Manually compute attention for Flax
    print("\n" + "="*80)
    print("FLAX ATTENTION COMPUTATION")
    print("="*80)
    scores_flax = compare_attention_scores(q_flax, k_flax)
    weights_flax = compare_attention_weights(scores_flax)
    attn_output_flax = compare_attention_output(weights_flax, v_flax)
    
    # Get full outputs
    print("\n" + "="*80)
    print("FULL MODEL OUTPUTS")
    print("="*80)
    
    out_te = model_te.apply(vars_te_modified, inputs_q, inputs_kv)
    out_flax = model_flax.apply(vars_flax, inputs_q, inputs_kv)
    
    print(f"\nTE output shape: {out_te.shape}")
    print(f"Flax output shape: {out_flax.shape}")
    print(f"\nTE output mean: {jnp.mean(out_te):.6f}")
    print(f"Flax output mean: {jnp.mean(out_flax):.6f}")
    print(f"\nTE output std: {jnp.std(out_te):.6f}")
    print(f"Flax output std: {jnp.std(out_flax):.6f}")
    
    diff = jnp.abs(out_te - out_flax)
    print(f"\n" + "="*80)
    print("DIFFERENCE ANALYSIS")
    print("="*80)
    print(f"Max absolute difference: {jnp.max(diff):.6e}")
    print(f"Mean absolute difference: {jnp.mean(diff):.6e}")
    print(f"Median absolute difference: {jnp.median(diff):.6e}")
    print(f"95th percentile difference: {jnp.percentile(diff, 95):.6e}")
    
    # Check where differences are largest
    max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    print(f"\nLargest difference at index: {max_diff_idx}")
    print(f"TE value: {out_te[max_diff_idx]:.6f}")
    print(f"Flax value: {out_flax[max_diff_idx]:.6f}")
    
    # Statistical test
    rtol, atol = 1e-5, 1e-5
    is_close = jnp.allclose(out_te, out_flax, rtol=rtol, atol=atol)
    print(f"\n" + "="*80)
    print(f"Are outputs close (rtol={rtol}, atol={atol})? {is_close}")
    print("="*80)
    
    if not is_close:
        print("\n⚠️  OUTPUTS ARE DIFFERENT!")
        print("\nPossible reasons:")
        print("1. TransformerEngine DotProductAttention uses different scaling")
        print("2. Different attention computation order (numerical precision)")
        print("3. Different mask handling")
        print("4. Different dropout implementation (even with deterministic=True)")
        print("5. TE might use fused kernels with different numerical behavior")
    
    return {
        'out_te': out_te,
        'out_flax': out_flax,
        'q_te': q_te,
        'q_flax': q_flax,
        'k_te': k_te,
        'k_flax': k_flax,
        'v_te': v_te,
        'v_flax': v_flax,
        'diff': diff
    }


def test_with_mask():
    """Test with causal mask to see if masking differs."""
    print("\n\n" + "="*80)
    print("TESTING WITH CAUSAL MASK")
    print("="*80)
    
    if not TE_AVAILABLE:
        print("TransformerEngine not available.")
        return
    
    batch_size, seq_len, dim = 2, 8, 64
    num_heads = 4
    seed = 42
    
    rng = jax.random.PRNGKey(seed)
    rng_input, rng_init = jax.random.split(rng)
    
    inputs_q = jax.random.normal(rng_input, (batch_size, seq_len, dim))
    inputs_kv = inputs_q
    
    # Create causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.expand_dims(mask, axis=(0, 1))  # [1, 1, seq_len, seq_len]
    mask = mask.astype(bool)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask (first head, first batch):\n{mask[0, 0]}")
    
    kernel_init = nn.initializers.xavier_uniform()
    
    model_te = MultiHeadDotProductAttentionTE(
        num_heads=num_heads,
        kernel_init=kernel_init,
        deterministic=True
    )
    
    model_flax = MultiHeadDotProductAttentionFlax(
        num_heads=num_heads,
        kernel_init=kernel_init,
        deterministic=True
    )
    
    vars_te = model_te.init(rng_init, inputs_q, inputs_kv, mask=mask)
    vars_flax = model_flax.init(rng_init, inputs_q, inputs_kv, mask=mask)
    
    # Copy parameters
    vars_te_modified = {
        'params': {
            'query': vars_flax['params']['query'],
            'key': vars_flax['params']['key'],
            'value': vars_flax['params']['value'],
            'out': vars_flax['params']['out']
        }
    }
    
    out_te = model_te.apply(vars_te_modified, inputs_q, inputs_kv, mask=mask)
    out_flax = model_flax.apply(vars_flax, inputs_q, inputs_kv, mask=mask)
    
    diff = jnp.abs(out_te - out_flax)
    print(f"\nWith mask - Max absolute difference: {jnp.max(diff):.6e}")
    print(f"With mask - Mean absolute difference: {jnp.mean(diff):.6e}")
    print(f"Are outputs close? {jnp.allclose(out_te, out_flax, rtol=1e-5, atol=1e-5)}")


def test_different_seq_lengths():
    """Test with different sequence lengths."""
    print("\n\n" + "="*80)
    print("TESTING WITH DIFFERENT SEQUENCE LENGTHS")
    print("="*80)
    
    if not TE_AVAILABLE:
        print("TransformerEngine not available.")
        return
    
    seq_lengths = [4, 8, 16, 32]
    dim = 64
    num_heads = 4
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")
        
        rng = jax.random.PRNGKey(42)
        inputs = jax.random.normal(rng, (2, seq_len, dim))
        
        kernel_init = nn.initializers.xavier_uniform()
        
        model_te = MultiHeadDotProductAttentionTE(
            num_heads=num_heads,
            kernel_init=kernel_init,
            deterministic=True
        )
        
        model_flax = MultiHeadDotProductAttentionFlax(
            num_heads=num_heads,
            kernel_init=kernel_init,
            deterministic=True
        )
        
        vars_te = model_te.init(rng, inputs, inputs)
        vars_flax = model_flax.init(rng, inputs, inputs)
        
        vars_te_modified = {
            'params': {
                'query': vars_flax['params']['query'],
                'key': vars_flax['params']['key'],
                'value': vars_flax['params']['value'],
                'out': vars_flax['params']['out']
            }
        }
        
        out_te = model_te.apply(vars_te_modified, inputs, inputs)
        out_flax = model_flax.apply(vars_flax, inputs, inputs)
        
        diff = jnp.abs(out_te - out_flax)
        print(f"Max diff: {jnp.max(diff):.6e}, Mean diff: {jnp.mean(diff):.6e}")


if __name__ == "__main__":
    results = detailed_comparison()
    test_with_mask()
    test_different_seq_lengths()