# Explanations

## Multiview3D — Detailed Explanation

This section explains the `Multiview3D` base class (defined in `slam3r/models.py`) step-by-step. `Multiview3D` implements the shared backbone used by specialized models (for example `Image2PointsModel` and `Local2WorldModel`), providing:

- patch embedding and image tokenization
- positional embeddings (cosine or RoPE)
- transformer encoder
- symmetric multiview transformer decoder
- downstream head plumbing and utility helpers

### Purpose & design

- `Multiview3D` is the core multi-view transformer backbone. It is designed to encode an arbitrary number of input views into patch tokens, exchange information across views in a paired (reference / source) decoder setting, and hand the decoder outputs to configurable heads that produce spatial outputs (depth/3D points and confidence maps).

### Initialization (\_\_init\_\_)

- Key responsibilities performed in `__init__`:
  - Validate and set input `img_size` and `patch_size`.
  - Create a `patch_embed` module via `get_patch_embed(...)` — this converts images to patch tokens.
  - Set positional embeddings through `_set_pos_embed`, supporting either 2D sin-cos (`'cosine'`) or RoPE (`'RoPE...'`) positional encodings.
  - Create encoder transformer blocks (if `need_encoder=True`) using `_set_encoder` and a final encoder norm (`enc_norm`).
  - Create two symmetric multiview decoder block lists via `_set_decoder` (`mv_dec_blocks1` and `mv_dec_blocks2`) and a `decoder_embed` linear projection from encoder dim -> decoder dim.
  - Create downstream heads with `_set_downstream_head`, producing `self.head1` and `self.head2` wrappers used by subclasses.
  - Apply an optional freeze strategy via `set_freeze`.

### Patch embedding: `_set_patch_embed`

- `patch_embed` is an object (selected by `patch_embed_cls`) that turns an image `(B, 3, H, W)` into patch tokens `(B, S, D_enc)` and (optionally) token positions.
- `self.patch_size` and `self.patch_embed.num_patches` are stored for further shape calculations.

### Encoder setup: `_set_encoder`

- Builds `self.enc_blocks`, a `ModuleList` of `Block` transformer layers with `enc_depth` layers.
- Sets `self.enc_norm` (LayerNorm) for post-encoder normalization.

### Positional embeddings: `_set_pos_embed`

- Two modes supported:
  - `'cosine'`: precomputed 2D sin-cos embeddings are stored in `enc_pos_embed` and `dec_pos_embed` buffers.
  - `'RoPE{freq}'`: rotary positional embeddings via `RoPE2D` (requires cuRoPE2D; raises ImportError if unavailable).
- If `RoPE` is used, explicit positional buffers are not used; instead, `rope` is provided to transformer blocks.

### Decoder setup: `_set_decoder`

- `self.decoder_embed`: linear projection mapping encoder features (D_enc) to decoder dimension (D_dec).
- Two symmetric decoder paths are created:
  - `mv_dec_blocks1`: used for the reference-side updates.
  - `mv_dec_blocks2`: used for the source-side updates.
- Each decoder block receives parameters to optionally normalize its memory (`norm_mem`) and a RoPE instance when needed.
- A final `self.dec_norm` LayerNorm is used to normalize the final decoder outputs.

### Downstream head: `_set_downstream_head` and `_downstream_head`

- Builds two downstream head modules using `head_factory`, `downstream_head1` and `downstream_head2`, configured by `head_type` and `output_mode`.
- Wraps heads via `transpose_to_landscape(...)` into `self.head1` and `self.head2` (this wrapper adjusts data layout depending on the head implementation).
- `_downstream_head(self, head_num, decout, img_shape)` is a thin helper that selects `head1` or `head2` and runs the head on `decout` — a list of decoder outputs across depth.

### Checkpoint compatibility: `load_state_dict`

- `Multiview3D.load_state_dict` provides compatibility shims between different checkpoint key naming conventions (`slam3r`, `croco`, `dust3r`).
- If loaded checkpoint lacks encoder weights and `need_encoder` is False, encoder keys are filtered out.
- The method renames `dec_blocks`/`dec_blocks2` keys to `mv_dec_blocks1`/`mv_dec_blocks2` to match this repo's naming.

### Freezing utilities: `set_freeze`

- Supports `'none'` (default) and `'encoder'` (freeze patch embed + encoder blocks).
- A special `'corr_score_head_only'` option was added for downstream models that attach a correlation head — it freezes all parameters except the correlation head norm and projection.

### Image encoding helpers: `_encode_image` and `_encode_multiview`

- `_encode_image(image, true_shape, normalize=True)`:
  - Applies `self.patch_embed` to produce `(B, S, D_enc)` tokens and token positions.
  - Runs tokens through encoder blocks `self.enc_blocks` and `self.enc_norm` if `normalize=True`.
- `_encode_multiview(views, view_batchsize=None, normalize=True, silent=True)`:
  - Encodes a list of view dicts into three lists (one per view): `shapes` (B,2), `enc_feats` (B,S,D_enc), `poses` (B,S,2).
  - Supports precomputed `img_tokens` in `views` (skips encoding) — those tokens will be normalized if `normalize=True`.
  - When many views are provided, views can be encoded in chunks controlled by `enc_minibatch`.
  - Returns `res_shapes, res_feats, res_poses` (each a Python list with `len(views)` elements).

### Multiview decoder: `_decode_multiview`

- Inputs:
  - `ref_feats` shaped `(R, B, S, D_enc)` and `src_feats` shaped `(V-R, B, S, D_enc)` (these are stacked over views).
  - `ref_poses` and `src_poses` with token positions.
  - Optional point-embedding tokens `ref_pes` / `src_pes` (used by `Local2WorldModel`).
- Steps:
  - Append the projected tokens `self.decoder_embed(...)` to `final_refs` and `final_srcs`.
  - Create relative-id arrays to describe how many cross-view memories each block should attend to.
  - For each decoder layer (loop over `self.dec_depth`):
    - Build `ref_inputs` and `src_inputs`, optionally adding point-embeddings.
    - Apply the corresponding `ref_blk` and `src_blk` blocks (from `mv_dec_blocks1` and `mv_dec_blocks2`) which implement cross-view attention between ref and src tokens.
    - Append the outputs to `final_refs` and `final_srcs`.
  - After the loop, remove the duplicate projected-entry stored at index 1, apply final `dec_norm` to last outputs.
  - Reshape each depth entry from `(R, B, S, D)` -> `(R*B, S, D)` and `(V-R, B, S, D)` -> `((V-R)*B, S, D)` so downstream heads can process per-sample stacks.
- Returns two lists: `final_refs` and `final_srcs` where each list has `dec_depth` entries (one per stage) and each entry is shaped for head consumption.

### View utilities: `split_stack_ref_src`

- Small helper used extensively by subclasses. Given a Python list of per-view tensors and lists of indices, it selects and stacks the reference and source elements into tensors shaped `(R, B, S, D)` and `(V-R, B, S, D)`.

### Practical notes & tips

- `Multiview3D` separates concerns: encoding (shared across views), decoding (cross-view attention), and heads (task-specific outputs). Subclasses normally only decide how to prepare inputs for the decoder and how to interpret head outputs.
- The stacked shapes `(R, B, S, D)` vs `(R*B, S, D)` can be confusing — keep track of when the code stacks by view (R dimension) and when it flattens to a per-sample batch for the heads.
- The symmetric decoder design (`mv_dec_blocks1` / `mv_dec_blocks2`) gives flexibility: reference-side blocks and source-side blocks are separate modules but generally share the same block class, allowing different internal behavior (for instance different `norm_mem` settings).
- When using RoPE positional encoding, ensure `RoPE2D` is installed (cuRoPE) or the code will raise an ImportError.

## Image2PointsModel — Detailed Explanation

This section explains the `Image2PointsModel` class (defined in `slam3r/models.py`) step-by-step. `Image2PointsModel` inherits from `Multiview3D` and implements a multiview image-to-3D-pointmap model with an optional correlation (retrieval) head.

- **Purpose:** Take multiple views (images or precomputed image tokens) and predict 3D point maps and confidence maps. One view acts as the reference (defines the output coordinate system) while the others are source views.

### High-level flow

- Encode each view into patch tokens using the shared patch embedding and encoder.
- Split tokens into reference and source groups (the model assumes a single reference in Image2PointsModel).
- Run a multiview decoder to let reference and source tokens interact.
- Pass decoder outputs to downstream heads to produce `pts3d` and `conf` maps.
- Optionally compute a shallow correlation score for source patches.

### Initialization (\_\_init\_\_)

- Calls `Multiview3D.__init__` to build patch embeddings, positional encodings, encoder, decoder and downstream heads.
- Adds correlation (retrieval) head parameters used by `get_corr_score` and optionally returned from `forward`:
- `self.corr_score_depth`: how many decoder blocks to run for the shallow correlation output (default provided by argument).
- `self.corr_score_norm`: a `LayerNorm` applied to the decoder outputs used by the correlation head.
- `self.corr_score_proj`: an MLP that projects decoder token features to a single scalar per token (patch score).

### Shapes and naming conventions (useful when reading code)

- V: number of views in the input list.
- R: number of reference views (for Image2PointsModel, R == 1).
- S: number of image patches per view (H/patch_size * W/patch_size).
- B: batch size (number of images per view in the mini-batch).
- Encoder token shape: (B, S, D_enc). After stacking over views, reference-group shape becomes (R, B, S, D_enc) and source-group shape becomes (V-R, B, S, D_enc).
- Decoder token shape after projection: (R, B, S, D_dec) and ((V-R), B, S, D_dec). Many methods later reshape these to (R*B, S, D_dec) or ((V-R)*B, S, D_dec).

### Encoding: `_encode_multiview` (used by both `get_corr_score` and `forward`)

- Accepts `views`, a list of dicts. Each dict typically contains at least `img` (tensor of shape `(B, 3, H, W)`) and `true_shape` (tensor `(B,2)`). If `img_tokens` are already provided, the method uses them directly.
- Embeds each image using `self.patch_embed` (returns tokens and token positions).
- Runs the encoder transformer blocks (unless tokens are precomputed).
- Returns three lists (one element per view): `shapes` (B,2), `enc_feats` (B,S,D_enc), and `poses` (B,S,2).
- Note: `enc_minibatch` controls chunking of views during encoding when many views are provided.

### Splitting reference and source tokens: `split_stack_ref_src`

- Given lists for all views and `ref_ids`/`src_ids`, this utility stacks the selected views into tensors shaped `(R, B, S, D)` and `(V-R, B, S, D)`.

### Multiview decoding: `_decode_multiview`

- Takes stacked reference and source tokens and lets them exchange information through symmetric decoder block lists `mv_dec_blocks1` (for reference side) and `mv_dec_blocks2` (for source side).
- The code projects encoder tokens into decoder dimensionality with `self.decoder_embed` before entering blocks.
- Decoder blocks run for `self.dec_depth` iterations. Each iteration produces updated tensors appended to `final_refs` and `final_srcs`.
- After finishing, the implementation removes duplicate intermediate elements (the initial projected entry), applies final `dec_norm`, and reshapes each entry from `(R, B, S, D)` into `(R*B, S, D)` and `(V-R, B, S, D)` into `((V-R)*B, S, D)`.
- Returns two lists: `final_refs` and `final_srcs` with one item per decoder depth level (useful for multi-scale heads).

### Correlation score head: `get_corr_score`

- Purpose: compute a patch-level correlation / retrieval score between the reference and each source view using only a few shallow decoder blocks.
- Steps:
- Calls `_encode_multiview` to get encoder tokens and poses.
- Uses `split_stack_ref_src` to form `ref_feats` and `src_feats` stacked tensors (R=1 expected).
- Projects encoder tokens to decoder dimensionality using `self.decoder_embed`.
- Runs a loop over `depth` decoder blocks (default `self.corr_score_depth`). In each iteration the code calls the same decoder block classes used in the full decoder, but only collects outputs for the source side. For the reference side it runs blocks only when needed to supply memory to the source-side blocks.
- After the shallow decoding, the code normalizes source-side tokens with `self.corr_score_norm` and projects each token to a scalar using `self.corr_score_proj` (an `Mlp` with output size 1). The resulting shape is `(num_src, B, S)`.
- The projection output is passed through `reg_dense_conf` (the same postprocessing used for confidence maps) with the model's `conf_mode` to produce stabilized correlation scores.
- Returns: `patch_corr_scores` shaped `(num_src, B, S)`. When the full `forward` returns a `pseudo_conf`, the shallow scores are reshaped and split per source view.

### Forward pass: `forward(self, views:list, ref_id, return_corr_score=False)`

- Args:
- `views`: list of view dicts (each with `img` or `img_tokens` and `true_shape`).
- `ref_id`: index of the reference view within `views`. `Image2PointsModel` assumes one reference view: `ref_ids = [ref_id]`.
- `return_corr_score`: if True, compute and return the shallow patch correlation scores for source views.
- Steps:
  1. Encode all views with `_encode_multiview` (returns `shapes`, `enc_feats`, `poses`).
  2. Split stacks into `ref_feats`, `src_feats` and their poses and shapes using `split_stack_ref_src`.
  3. Call `_decode_multiview(ref_feats, src_feats, ...)` to get `dec_feats_ref` and `dec_feats_src`. Each is a list with `dec_depth` entries. Each entry has shape `(R*B, S, D_dec)` for refs and `((V-R)*B, S, D_dec)` for srcs.
  4. Run downstream heads: `self._downstream_head(1, dec_feats_ref, ref_shapes.reshape(-1,2))` and `self._downstream_head(2, dec_feats_src, src_shapes.reshape(-1,2))`. Heads map token lists to spatial outputs such as `pts3d` and `conf`.
  5. If `return_corr_score` is True, compute shallow correlation on `dec_feats_src[self.corr_score_depth]` using `self.corr_score_norm` and `self.corr_score_proj` (same as `get_corr_score` but run within forward's context).
  6. Split the outputs back to per-view `results`: iterate through the input `views` and fill a dict per view. For the reference view the model returns:
  - `'pts3d'`: the predicted 3D pointmap in the reference coordinate system (shape `(B, H, W, 3)` depending on head config).
  - `'conf'`: the confidence map for the reference prediction.
    For source views the model returns:
  - `'pts3d_in_other_view'`: predicted 3D pointmap for the source view expressed in the reference frame.
  - `'conf'`: the source confidence map.
  - optionally `'pseudo_conf'`: per-patch shallow correlation scores when `return_corr_score=True`.
- Returns: `results`, a list with one dict per input view.

### Practical notes & tips

- `Image2PointsModel` is implemented as a specialization of `Multiview3D`. The heavy lifting (patch embedding, encoder, decoder, positional encodings, and head plumbing) lives in the `Multiview3D` base class.
- The code uses the convention of stacking `R` reference views and `V-R` source views into tensors for efficient batch transformer processing. The model frequently reshapes between `(R, B, S, D)` and `(R*B, S, D)` to feed downstream heads that expect the latter.
- `corr_score_depth` controls how deep the shallow decoder runs when computing correlation—lower values make the retrieval computation cheaper and emphasize more local similarity, while larger values allow more cross-view context.
- `reg_dense_conf` is applied to the raw scalar projections to make them numerically stable / interpretable as confidences.

## Local2WorldModel — Detailed Explanation

This section explains the `Local2WorldModel` class (defined in `slam3r/models.py`) step-by-step. `Local2WorldModel` extends `Multiview3D` to accept input 3D pointmaps for some views and refine / transform them across views.

### Purpose

- Accepts multiple reference views (object- or world-centric frames) and multiple source views (camera/keyframes) and does two main tasks:
  - refine the input 3D pointmaps of the reference views, and
  - transform the input 3D pointmaps of the source views into the coordinate systems of the reference views.

### Initialization (`__init__`) and helpers

- Calls `Multiview3D.__init__` to set up patch embedding, encoder, decoder and heads.
- Sets `self.dec_embed_dim` from `self.decoder_embed.out_features` (decoder token dimension).
- Creates a `void_pe_token` parameter used when point patches are masked out.
- Adds a simple pointmap embedder via `set_pointmap_embedder()`:
  - `self.ponit_embedder` (note the repository's variable name) is a `nn.Conv2d(3, dec_embed_dim, kernel_size=patch_size, stride=patch_size)` that maps 3-channel pointmaps (H, W, 3) into patch tokens `(B, S, D_dec)`.

### Point-embedding: `get_pe`

- Purpose: convert input 3D pointmaps (either world or camera coords) into decoder-space tokens that can be added to decoder inputs.
- Input views:
  - For reference views, expects `view['pts3d_world']` (shape `(B, H, W, 3)` or already `(B, 3, H, W)`).
  - For source views, expects `view['pts3d_cam']`.
- For each view, `get_pe`:
  - permutes the pointmap to channel-first if needed,
  - applies the conv `ponit_embedder` to produce `(B, dec_embed_dim, H', W')`,
  - permutes and reshapes to `(B, S, D_dec)` where `S = H'*W'` (patch tokens),
  - if `patch_mask` exists in the view, uses it to replace masked patch embeddings with `self.void_pe_token`.
- Returns a Python list `pes` with one `(B, S, D_dec)` element per input view.

### Forward pass (`forward(self, views:list, ref_ids=0)`)

- Args:
  - `views`: list of dicts. Each view typically includes `img` (or precomputed `img_tokens`), `true_shape`, and additionally either `pts3d_world` (for reference) or `pts3d_cam` (for source).
  - `ref_ids`: index or list of indices identifying reference views.

- Steps (high-level):
  1. Validate `ref_ids` and build `src_ids` as the complement.
  2. Encode all views with `_encode_multiview(views)` to obtain `shapes`, `enc_feats`, and `poses`.
  3. Compute point-embeddings `pes = self.get_pe(views, ref_ids=ref_ids)`.
  4. Use `split_stack_ref_src` to produce stacked tensors for `ref_feats`, `src_feats`, `ref_poses`, `src_poses`, and importantly `ref_pes`, `src_pes` (shapes also split for heads).
  5. Call `_decode_multiview(ref_feats, src_feats, ref_poses, src_poses, ref_pes, src_pes)` — note: point-embeddings are supplied to the decoder so blocks receive both image token features and pointmap tokens.
  6. Run downstream heads through `_downstream_head(1, dec_feats_ref, ref_shapes)` and `_downstream_head(2, dec_feats_src, src_shapes)` to produce `pts3d` and `conf` outputs.
  7. Split head outputs back to per-view dicts and return the list of results.

### Important shapes & behavior

- `pes` elements are `(B, S, D_dec)` (after conv + reshape) matching decoder dimension.
- During `_decode_multiview`, reference and source token inputs are formed as `final_refs[-1] + ref_pes` and `final_srcs[-1] + src_pes` when `ref_pes`/`src_pes` are provided — this effectively injects pointmap information as additive positional/content tokens into the decoder pipeline.
- Downstream heads consume the decoder outputs shaped `(R*B, S, D_dec)` and `((V-R)*B, S, D_dec)` to generate spatial predictions. The code then reshapes/slices these per original view.

### Masking and the `void_pe_token`

- If a view contains a `patch_mask`, `get_pe` will place `self.void_pe_token` into positions corresponding to masked patches. This allows the decoder to learn a neutral embedding for missing or invalid point patches instead of using zeros which could bias attention.

### Practical notes & tips

- `Local2WorldModel` is tailored for tasks where per-view 3D pointmaps are available at input (e.g., SLAM outputs, depth fusion) and need to be refined jointly. Injecting `pes` into the decoder tightly couples image features and pointmap features in the cross-view attention updates.
- Because the point embeddings are a simple conv, they share the same patch-granularity as image tokens. That makes addition `final_refs[-1] + ref_pes` a natural operation without extra reshaping.
- When using multiple reference views (`ref_ids` length > 1), the returned `res_ref['pts3d']` contains stacked outputs for all reference views and the splitting logic divides them by `B` accordingly.

### Example usage (pseudo-code)

- Prepare `views` with `img`, `true_shape`, and for reference views `pts3d_world` (and for sources `pts3d_cam`). Optionally add `patch_mask` for missing point patches.
- `model = Local2WorldModel(...)`
- `results = model(views, ref_ids=[0,1])`
- `results[i]['pts3d']` (if `i` is a reference id) or `results[i]['pts3d_in_other_view']` (if source) are available per view.

If you want, I can also add a short diagram or inline code comments in `slam3r/models.py` to point readers to `get_pe`, `set_pointmap_embedder`, and where `ref_pes`/`src_pes` are added to the decoder. Would you like inline comments or a diagram added?
