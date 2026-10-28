# Sample Model Distribution

## Model paths (not included)

The Asset Store sample includes inference manifests but no model weights, graphs, tokenizer vocabularies, or label embeddings. After obtaining a model under its own terms, install it at a path relative to `Application.streamingAssetsPath` or use the example's model directory/download flow.

| Family | Configured path prefix | Files expected from the user |
| --- | --- | --- |
| Clip | `Clip/mobileclip_s0_export` | NCNN image encoder, tokenizer data, label embeddings |
| CodeFormer | `CodeFormer/models` | Encoder, generator, face detector NCNN pairs |
| DeepFillV2 | `DeepFileV2` | NCNN `.param` + `.bin` pair by default; source ONNX + `.param` is an alternate representation |
| Matting | `Matting` | NCNN pair |
| RealESRGAN | `RealESRGAN/models` | `realesr-animevideov3-x4` default NCNN pair; other variants on demand |
| YOLO | `Yolo` | YOLOv8n-seg NCNN pair; YOLO11 assets are not published yet |

## Excluded payloads

Do not put model files for any family in this Asset Store package or its samples. Runner configuration may name an external model location, but consumers must obtain the model separately under its own license.

## Configured GitHub Release pages

`AIImageModelDelivery` downloads individual files from [newzjh/Aexis Releases](https://github.com/newzjh/Aexis/releases). Each file is mapped to its existing release tag and installed at the corresponding path below `Application.persistentDataPath/AIImageModels`.

| Group | Release tag | Available files |
| --- | --- | --- |
| CLIP MobileCLIP S0 | [`clip`](https://github.com/newzjh/Aexis/releases/tag/clip) | Tokenizer files, label embeddings, and image encoder pair |
| CodeFormer | [`codeformer`](https://github.com/newzjh/Aexis/releases/tag/codeformer) | Encoder, generator, and YOLOv7-lite-e detector pairs |
| Matting | [`matting`](https://github.com/newzjh/Aexis/releases/tag/matting) | `matting.param`, `matting.bin` |
| YOLOv8 person segmentation | [`yolo`](https://github.com/newzjh/Aexis/releases/tag/yolo) | `yolov8n_seg.ncnn.param`, `yolov8n_seg.ncnn.bin` |
| Real-ESRGAN | [`realesr`](https://github.com/newzjh/Aexis/releases/tag/realesr) | Configured NCNN variants |
| GFPGAN | [`gfpgan`](https://github.com/newzjh/Aexis/releases/tag/gfpgan) | Configured NCNN files |
| DeepFillV2 | [`DeepFileV2`](https://github.com/newzjh/Aexis/releases/tag/DeepFileV2) | Case1 and HiFill files |
| Qwen3.5 mobile Q4/Q8 | [`qwen3.5_0.8b_mobile_q4`](https://github.com/newzjh/Aexis/releases/tag/qwen3.5_0.8b_mobile_q4), [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/Aexis/releases/tag/qwen3.5_0.8b_mobile_q8) | Model config and weight shards |
| Stable Diffusion inpainting | [`model`](https://github.com/newzjh/Aexis/releases/tag/model), [`sdinpainting`](https://github.com/newzjh/Aexis/releases/tag/sdinpainting) | Text encoder, VAE, and UNet files |

YOLO11 and Qwen3.5 full-precision are not downloadable from the current releases because their complete required file sets are not present. The model-download UI marks those groups unavailable instead of constructing a URL for a missing archive.

## Release gate

Before publishing any separate archive that contains a model artifact, complete an auditable record for every file: upstream project and URL, immutable revision or checksum, upstream license text, copyright notice, modification/export steps, and redistribution approval. Keep this record with the release tag. The Asset Store EULA for Aexis does not grant rights to model weights.

Do not infer licenses for files from their model family names. The Real-ESRGAN license previously carried next to a sample model has been removed along with that model. Include applicable notices with any separately distributed model archive after its own provenance review.

## Reduced Main2 delivery

The reduced Main2 release policy is defined by `AIImageModelDelivery`. A local build may bundle only complete model groups already installed by the developer. The Asset Store package does not supply those groups. MONAI and VISTA are never bundled into the reduced release catalog; their external Release pages above do not make medical input or data part of this package.

Runtime and editor downloads use the existing per-release flat assets and explicit file-to-path
mappings. Qwen3.5 Q4 shard paths are read from its release manifest so newly added parts are
downloaded without hard-coding a part count. Downloads install below the persistent model directory
without modifying project `StreamingAssets`.

On Android, each bundled model group is streamed from APK assets into that persistent model root
when first used, because the runners require ordinary file paths. If preparation cannot complete,
the runtime download dialog offers the same mapped GitHub Release files.
