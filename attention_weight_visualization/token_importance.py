import torch

def token_importance_from_attention(
    attention,
    layer='last',   # 'last' | 'mean'
    head='mean'     # 'mean' | int
):

    if isinstance(attention, tuple):
        attn = torch.stack(attention)
    else:
        attn = atttention

    # attention: (L, B, H, S, S)
    attn = attn[:,0]  # remove batch

    if head == 'mean':
        attn = attn.mean(dim=1)
    else:
        attn = attn[:, head]

    if layer == 'last':
        attn = attn[-1]
    else:
        attn = attn.mean(dim=0)

    token_scores = attn.sum(dim=0)
    token_scores = token_scores / token_scores.max()

    return token_scores

