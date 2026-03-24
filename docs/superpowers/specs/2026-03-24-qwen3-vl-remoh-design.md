# Qwen3-VL REMOH Migration Design

Date: 2026-03-24

## Goal

Migrate the current PVChat personalized video-chat training stack from the InternVideo2-based architecture to native Qwen3-VL while preserving the core PVChat training intent:

- keep native Qwen3-VL as the backbone
- preserve personalized `<sks>` token learning
- preserve REMOH-style head routing as a lightweight adaptation mechanism
- minimize disturbance to the pretrained visual encoder and the pretrained language model
- retain a two-stage training schedule

This design applies to training-time architecture only. It is separate from the already-updated QA data generation pipeline.

## Current PVChat Behavior

The current PVChat training stack is built around:

- `InternVideo2` vision encoder
- `Q-former` bridge
- `Mistral-7B` language model

In the current implementation:

- video features are extracted by the vision encoder
- `Q-former` uses learned query tokens to read those visual features
- the resulting bridge tokens are projected into the LLM hidden size
- those projected visual tokens are inserted into the LLM input sequence
- REMOH is attached only inside the `Q-former` attention layers

This means PVChat does not modify the full language-model reasoning path directly. It modifies the visual-to-language bridge.

## Qwen3-VL Constraint

Native Qwen3-VL does not expose a `Q-former`-style transformer bridge. Its multimodal path is:

- `Qwen3VLVisionModel`
- visual token compression / merger
- replacement of `<video>` placeholder embeddings with visual embeddings
- `Qwen3VLTextModel`

Important implications:

- `merger` and `deepstack_merger_list` are not transformer blocks
- they do not contain multi-head attention
- therefore the current REMOH implementation cannot be transplanted there directly

The closest native transformer region to the old `Q-former bridge` is the earliest text decoder layers, because:

- they are the first transformer layers that operate on the joint text-plus-visual token sequence
- they also receive DeepStack visual enhancement
- they behave as the first multimodal fusion layers after visual token injection

## Final Architecture Decision

Use native Qwen3-VL and place REMOH only in the earliest multimodal fusion layers:

- backbone: native `Qwen3-VL`
- REMOH location: `Qwen3VLTextDecoderLayer[0:3].self_attn`
- visual tower: frozen in stage 1
- visual merger and deepstack merger: frozen in both stages
- later text decoder layers: frozen in both stages unless a later experiment explicitly changes this

This is the closest native-Qwen equivalent to the old PVChat design principle of "modify the bridge, not the whole LLM".

## Personalized Tokens

Add the following special tokens to the Qwen tokenizer:

- `<sks>`
- `<sks_token1>` through `<sks_token16>`

After tokenizer extension:

- resize input embeddings
- resize `lm_head`
- preserve all original token rows
- train only the rows corresponding to the new personalized tokens

This mirrors the effective behavior used in YoLLaVA personalization and avoids unintended drift in the base vocabulary.

## Trainable Parameters

### Stage 1

Train only:

- REMOH-added parameters inside `Qwen3VLTextDecoderLayer[0:3].self_attn`
- `embed_tokens` rows for `<sks>` and `<sks_token1:16>`
- `lm_head` rows for `<sks>` and `<sks_token1:16>`

Freeze:

- full visual tower
- `visual.merger`
- `visual.deepstack_merger_list`
- text decoder layers `3:`
- original weights of text decoder layers `0:3`, except the newly added REMOH parameters
- all non-personalized embedding rows
- all non-personalized `lm_head` rows

Initialization behavior:

- keep the current PVChat REMOH style
- use the same small-start behavior, not identity-at-init

### Stage 2

Continue training everything from stage 1, and additionally:

- apply LoRA to the last two visual transformer blocks only
- target modules:
  - `visual.blocks.25.attn.qkv`
  - `visual.blocks.25.attn.proj`
  - `visual.blocks.26.attn.qkv`
  - `visual.blocks.26.attn.proj`

Do not train:

- visual MLP sublayers
- earlier visual blocks
- merger modules
- later text decoder layers

This preserves the pretrained visual representation as much as possible while still allowing a small amount of visual-side adaptation in the final stage.

## Why This Mapping Is Preferred

Compared with placing REMOH in the visual tail:

- putting REMOH in the last visual blocks would directly alter visual feature extraction
- this conflicts with the goal of preserving the pretrained video encoder behavior as much as possible

Compared with placing REMOH across the whole text decoder:

- that would turn REMOH into a global LLM modification
- this drifts away from the original PVChat paper intent

Placing REMOH only in the first three Qwen text decoder layers is the best compromise:

- transformer-based
- head-level modification is possible
- located after visual encoding
- located before deeper language reasoning
- semantically closest to the old multimodal bridge

## Module Mapping

Old PVChat module to Qwen3-VL counterpart:

- InternVideo2 vision encoder -> `Qwen3VLVisionModel`
- Q-former bridge -> earliest multimodal fusion region in `Qwen3VLTextModel.layers[0:3]`
- REMOH in Q-former attention -> REMOH in `Qwen3VLTextDecoderLayer[0:3].self_attn`
- personalized query-token mechanism -> personalized tokenizer tokens and their embedding / output-head rows
- stage-2 partial visual adaptation -> LoRA on final two visual attention blocks

## Implementation Shape

The implementation should introduce a Qwen-specific training path rather than patch the old InternVideo2 model classes in place.

Expected code areas:

- a Qwen-based model wrapper or training entry that loads native `Qwen3VLForConditionalGeneration`
- tokenizer extension for `<sks>` and `<sks_token1:16>`
- selective optimization of only personalized embedding / head rows
- REMOH-enabled replacements for `Qwen3VLTextAttention` in the first three decoder layers
- stage-2 LoRA attachment for the last two visual attention blocks
- two-stage dataset scheduling preserved from PVChat

## Risks

- early Qwen text layers are technically decoder layers, not an explicit bridge module
- therefore this is an architectural approximation of the old Q-former role, not a literal one-to-one transplant
- row-only optimization for embeddings and `lm_head` requires careful post-step restoration logic
- REMOH on Qwen attention must preserve input/output tensor shapes exactly to remain generation-compatible

## Out of Scope

Not part of this design:

- replacing Qwen3-VL with InternVL3.5
- redesigning REMOH into a non-attention connector
- modifying data generation again
- changing the already approved emotion-QA expansion design

## Acceptance Criteria

The migration is considered architecturally correct when:

- native Qwen3-VL remains the backbone
- personalized tokens are added and can be generated
- REMOH is active only in the first three text decoder self-attention layers
- stage 1 does not update the visual tower
- stage 2 updates only LoRA adapters on the last two visual attention blocks
- non-personalized embedding and `lm_head` rows remain unchanged across optimization steps
