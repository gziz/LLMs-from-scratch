# Notes about GPT with KV Cache

Things you should know to fully grasp the KV Cache algorithm

### Original questions
How does it work?
- The definition of KV cache says that K and V are cached because they can be reused, but can you ellaborate?
    - It's called a KV cache because we are only caching the keys and the values, we are not caching the queries.
- Why are not we not caching the queries? Is it because they cannot be reused?
    - Even better, it's because after we have populated the caches, we only need the query from the last token generated!

## How it works
- For starters, KV cache is only possible during inferening, it's not possible during training.
- The KV cache is used within the MHA attention class.
- During inference, a `MHA.forward` with a KV cache implementation has two modes: prefill and decode.

We are going to explain each mode with a scenario in mind, hence, let's set some context about our scenario:
- We are inferencing.
- The system receives a user prompt, for which the model needs to generate tokens. 
    This initial user prompt (a.k.a user message) has a size of n tokens.
- As we generate tokens (the AI response) we are calling model.forward multiple times (at most max_tokens), from each call we obtain the next token generated.

### First model.forward() call
Prefill:
- This step occurs during the first `model.forward` pass. It's crucial as it populates the KV cache for the rest of the inferencing iterations. This step is not only for the KV cache population, because the MHA work is performed this step also serves to generate the token n+1.
- The input for the first `model.forward` is the tokenized initial prompt (composed of n tokens).

- Inside MHA.forward, the input is `x.size -> (B, curr_tokens, d_in)`. Where curr_tokens is n (length of initial prompt).
- The output of MHA.forward is `context_vec.size -> (B, curr_tokens, d_out)`
- Note the above two lines happened within the `MHA.forward` call of a single `TransformerBlock` layer.
- A `model.forward` call has multiple `TransformerBlock` layers, therefore each `MHA.forward` call within those layers, would be in prefill mode.
    - All the MHA within the `TransformerBlock` layers need to be in prefill mode because the output of `TransformerBlock` layer_1 is the input of layer_2.
        - Which means the output size of layer_1 must match the input size of layer_2.
    - This means we have a KV cache for each `TransformerBlock`!  
- The last `TransformerBlock` layer would return the output `x`. `GPTModel.forward` having `x`, would pass it through the output FFN that converts x from `(B, curr_tokens, M)` into `(B, curr_tokens, vocab_size)`. These are now the logits.
```
for blk in self.trf_blocks:
    x = blk(x, use_cache=use_cache)

x = self.final_norm(x)
logits = self.out_head(x)
```
- Note that from these logits a vector of vocab_size for each token (`n` vectors), however, we only need the last token to use arg_max with an generate token `n+1`.
```python
next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
```

### (2nd to max_tokens) model.forward calls
Decode: 
- Now that we have generated the token n+1 and we have populated the KV caches, we continue with the decoding step.
- Here's the magic: we only need to pass a single token (per batch) in order to perform attention.

- The input for subsequent `model.forward` calls is just the last generated token.
```python
logits = model(next_idx, use_cache=True)  # next_idx.shape -> (B, 1)
```

- Inside `MHA.forward`, the input is `x.size -> (B, 1, d_in)`. Only 1 token!
- We compute Q, K, V projections only for this single new token:
```python
keys_new = self.W_key(x)      # Shape: (B, 1, d_out)
values_new = self.W_value(x)  # Shape: (B, 1, d_out)
queries = self.W_query(x)     # Shape: (B, 1, d_out)
```

- The new keys and values are concatenated to the existing cache:
```python
self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)  # (B, n+1, NH, H_dim)
self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
keys, values = self.cache_k, self.cache_v
```

- Now the attention computation:
    - `queries` has shape `(B, NH, 1, H_dim)` — only the new token's query
    - `keys` has shape `(B, NH, n+1, H_dim)` — all tokens so far (from cache)
    - `values` has shape `(B, NH, n+1, H_dim)` — all tokens so far (from cache)

- The attention scores are computed as:
```python
attn_scores = queries @ keys.transpose(2, 3)
# (B, NH, 1, H_dim) @ (B, NH, H_dim, n+1) -> (B, NH, 1, n+1)
```

- The causal mask is adjusted using `ptr_current_pos` to select the correct row:
```python
mask_bool = self.mask.bool()[
    self.ptr_current_pos:self.ptr_current_pos + 1, :num_tokens_K
]
# This gives us a (1, n+1) mask for the new token position
```

- After attention, the context vector has shape `(B, 1, d_out)` — just one token's representation.
- The output of `MHA.forward` is `context_vec.size -> (B, 1, d_out)`
- This propagates through all `TransformerBlock` layers, each updating their own KV cache.
- Finally, logits have shape `(B, 1, vocab_size)`, and we take the argmax to get the next token.

### Why is this efficient?
- Without KV cache: For each new token, we recompute K and V for ALL previous tokens. Complexity: O(n²) per generation step.
- With KV cache: We only compute K and V for the NEW token, reusing cached values. Complexity: O(n) per generation step.
- The trade-off: We use more memory to store the cache, but save significant computation time.






### Follow up questions
During decoding, how is it that we only need the query from the last token?
- In a MHA without KV cache we pass all tokens from 1 to N. During the attention computation we have queries from tokens 1 to N.
- In a MHA with KV cache we pass the token N only. During the attention computation we have the query for token N only.
    - Eventhough we are only passing token N to the MHA, because of the kv cache, we have the keys and values for all tokens from 1 to N-1 in the cache.
    - However, we only have the query for token N, how is it that we only need this one?
    - Can you prove with math that having the queries for tokens 1 to N-1 is redundant?
- [Answer youtube vid](https://youtu.be/aD8OD4e0MIE)


