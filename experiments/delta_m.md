
### delta M
--
delta_M (6×6)

What it is: A learnable orthogonal 6×6 matrix that rotates the [cosx, cosy, cosz, sinx, siny, sinz] sin/cos components of RoPE, independently per frequency bin.

How it's built (base_denoise_actor.py:656):
1. camera_predictor maps cam_feat → 36 scalars → reshape to (B, 6, 6) as A_skew
2. Anti-symmetrize: A = A_skew - A_skew^T (guarantees skew-symmetric)
3. Norm-clip: Frobenius norm clamped to max_norm=3.0
4. delta_M = matrix_exp(A) → orthogonal 6×6 matrix

How it's applied (position_encodings.py:507):
feat = torch.einsum('bnci,bji->bncj', feat, delta_M)  # (B, N, d//6, 6)
The same 6×6 matrix mixes the 6 sin/cos components at each frequency bin independently.

---
delta_M_full (D×D)

What it is: Same concept but the matrix is D×D where D = (embedding_dim // 6) * 6 — it operates over the entire flattened sin/cos feature space.

How it's built (base_denoise_actor.py:644):
- Same skew-sym + norm-clip + matrix_exp pipeline, but camera_predictor outputs D*D scalars

How it's applied (position_encodings.py:511):
feat = torch.einsum('bni,bji->bnj', feat.reshape(B, N, -1), delta_M).reshape(B, N, nb, 6)
Flattens [B, N, d//6, 6] → [B, N, D] and applies the full D×D matrix — this allows cross-bin mixing, not just within-bin.

# Approach for VLA 
Now we want there to be a deltaM (per) image (i.e. each set of iamge tokens)


### how to use deltaM 
deltaM will be based on an extra register token used per image. This token comes from the thumbnail token from the VLM. 
Currently in the preprocessing of the VLM, the thumbnail token is NOT used, so we also need to store it.
Now once we have the thumbnail token, (after each image-self attention block) we will use the token to create a delta M matrix, then multiple that with the rope features before the next image-self attn block.

make sure the deltaM used for each image is different! 