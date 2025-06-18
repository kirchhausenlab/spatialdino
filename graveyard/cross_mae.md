## Cross-MAE

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads, ...):
        self.cross_attn = cross_attn_class(
            encoder_dim, decoder_dim, num_heads, ...
        )
```

## Weighted Feature Map

```python
class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        self.linear = nn.Linear(k, decoder_depth, bias=False)
```

## Reconstruction Loss

```python
class Projector(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: Tuple[int, int, int]):
        self.project = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
```

## Masked Loss implementation

```python
def forward_masked(self, pred_patches, target_patches, student_masks_flat, ...):
    loss = (pred_patches - target_patches) ** 2
    loss = loss.mean(dim=-1).flatten()
    loss = loss * masks_weight
    return loss.sum() / student_masks_flat.shape[0]
```
