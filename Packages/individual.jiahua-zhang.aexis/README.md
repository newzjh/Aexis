<p align="center">
  <a href="https://github.com/newzjh/Aexis">
    <img src="https://raw.githubusercontent.com/newzjh/Aexis/main/Documents/AssetStore/MarketingImages/aexis-icon.png" width="88" alt="Aexis on-device inference engine"/>
  </a>
</p>

<h1 align="center">Aexis Unity Package</h1>

<p align="center">
  <a href="https://github.com/newzjh/Aexis"><img src="https://img.shields.io/badge/GitHub-AIImage-181717?style=flat&logo=github" alt="GitHub"/></a>
  <a href="https://assetstore.unity.com/preview/397636/1435586"><img src="https://img.shields.io/badge/Unity%20Asset%20Store-Aexis-222C37?style=flat&logo=unity" alt="Unity Asset Store"/></a>
  <a href="https://github.com/sponsors/newzjh"><img src="https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-EA4AAA?style=flat&logo=githubsponsors" alt="Sponsor Aexis on GitHub"/></a>
  <a href="#community-links"><img src="https://img.shields.io/badge/LinkedIn-Coming%20soon-0A66C2?style=flat&logo=linkedin" alt="LinkedIn coming soon"/></a>
  <a href="#community-links"><img src="https://img.shields.io/badge/Discord-Coming%20soon-5865F2?style=flat&logo=discord" alt="Discord coming soon"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Asset%20Store%20EULA-3C8C6A?style=flat" alt="Asset Store EULA"/></a>
</p>

<p align="center"><sub><a href="#at-a-glance">At a glance</a> &bull; <a href="#install">Install</a> &bull; <a href="#runner-evidence">Examples</a> &bull; <a href="#tested-environments">Tested environments</a> &bull; <a href="#how-to-help">How to help</a> &bull; <a href="#license">License</a></sub></p>

`individual.jiahua-zhang.aexis` is the single Unity Package Manager (UPM) package for the Aexis on-device inference engine. AIImage is an application built on Aexis; it is not the engine namespace, a package dependency, or a runtime requirement.

## At a glance

- Unity 2022.3 LTS through Unity 6000.3 package baseline.
- Self-owned NCNN and ONNX import and execution paths with Pack4 RenderTexture storage.
- No Unity Sentis, ONNX Runtime, Tencent ncnn, MNN, UniTask, or native inference-plugin runtime dependency.
- Importable reusable runners and the full AIImage Main2 application example.

### Business inquiries

For commercial, integration, or partnership discussions, contact [newzjh@126.com](mailto:newzjh@126.com).

## How to help

- [Star Aexis on GitHub](https://github.com/newzjh/Aexis) and report reproducible package issues.
- Review the package on the [Unity Asset Store](https://assetstore.unity.com/preview/397636/1435586).
- [Sponsor on GitHub](https://github.com/sponsors/newzjh).
- See the repository [support guide](https://github.com/newzjh/Aexis/blob/main/SUPPORT.md) for current channels.

## Status and compatibility

| Item | Supported release baseline |
| --- | --- |
| Unity | 2022.3 LTS through Unity 6000.3 |
| Script runtime | Unity-supported .NET profile for the installed editor |
| GPU path | Compute-shader-capable graphics API with RenderTexture support |
| Package dependencies | Unity modules only; dynamic JSON is source-vendored in the optional application sample |
| Runtime namespaces | `Aexis`, `Aexis.Async`, `Aexis.Onnx`, `Aexis.Ncnn`, `Aexis.Execution` |

`Aexis.Async` exposes BCL `Task` and does not ship, vendor, or reference UniTask. This keeps the engine compatible with Unity 2022.3 and prevents package-level conflicts when a host project uses its own UniTask version or distribution.

## Runner evidence

AIImage Main2 is the complete application example built on Aexis. It contains runner configuration for Qwen3.5 Q4/Q8, CLIP, CodeFormer, Matting, Real-ESRGAN, GFPGAN, YOLO plus DeepFillV2, YOLO plus Stable Diffusion inpainting, and MONAI/VISTA. The observations below are historical validation records; their input and output images are not part of the Asset Store package.

### Qwen3.5 Mobile Q4/Q8 and CLIP MobileCLIP S0


Qwen Q4 passed a Windows strict-texture text smoke in 65.533 s using 48 cache textures. It also passed the Android MuMu Vulkan Main2 run in 326.821 s across 103 decoder steps, emitted 397 visible characters, and detected four people. Q8 passed a strict multimodal smoke in 71.618 s with six generated tokens. CLIP's successful 2026-07-23 score artifact ranked `Photo` at `0.332389` and `Portrait` at `0.265945`; its current strict CommandBuffer profile rejects an undeclared temporary RT at `transpose_121`.

### CodeFormer Face Restoration


The CodeFormer runner processed `ref/03.jpg`, detected one face, and completed in 17,958 ms (Windows/Vulkan, 2026-07-29).

### Foreground Matting


The matting runner completed at 360x202 in 1,103 ms with the strict texture plan and CommandBuffer path.

### GFPGAN Face Restoration


The GFPGAN runner processed `ref/03.jpg` in 6,053 ms with the strict Pack4 guard (Windows/Vulkan, 2026-07-29). The image is the raw output; this input visibly produces facial distortion, so it is execution evidence rather than a quality claim.

### YOLO Person Segmentation plus DeepFillV2


For `ref/deepfillv2/DeepFillv2-main/test_data/3.png`, YOLO found seven people with mask coverage `0.042730`. DeepFillV2 completed in 2,196 ms, including 1,995 ms inference (Windows/Vulkan, 2026-07-29). The result retains the actual residual removal artifacts.

### Real-ESRGAN AnimeVideo v3 x4


The CommandBuffer Pack4-only Real-ESRGAN validation processed `ref/03.jpg` in 1,057 ms (Windows/Vulkan, 2026-07-29).

### YOLO plus Stable Diffusion Inpainting


For `ref/deepfillv2/DeepFillv2-main/test_data/3.png`, YOLO found seven people with mask coverage `0.042730`; the 12-step Stable Diffusion inpainting run completed in 630,213 ms (Windows/Vulkan, 2026-07-29). It removes people more completely than the DeepFillV2 result, while retaining the actual residual objects and artifacts.

### MONAI and VISTA

> **Screenshot slot reserved:** A visual example requires reproducible, distributable medical input and confirmed data/model permissions. No medical image, private golden, checkpoint, or data result is published by this package.

## Release downloads

The Main2 model downloader fetches the published files from [newzjh/Aexis Releases](https://github.com/newzjh/Aexis/releases) by their release tag and exact asset name. CLIP, CodeFormer, Matting, YOLOv8, DeepFillV2, Real-ESRGAN, GFPGAN, and the published Qwen variants use their matching release assets. YOLO11 and Qwen full precision are shown as unavailable until complete release file sets are published. See [model distribution](Documentation~/model-distribution.md).

| Model group | Release download page |
| --- | --- |
| Qwen3.5 Q4 | [`qwen3.5_0.8b_mobile_q4`](https://github.com/newzjh/Aexis/releases/tag/qwen3.5_0.8b_mobile_q4) |
| Qwen3.5 Q8 | [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/Aexis/releases/tag/qwen3.5_0.8b_mobile_q8) |
| CLIP MobileCLIP S0 | [`clip`](https://github.com/newzjh/Aexis/releases/tag/clip) |
| CodeFormer | [`codeformer`](https://github.com/newzjh/Aexis/releases/tag/codeformer) |
| Matting | [`matting`](https://github.com/newzjh/Aexis/releases/tag/matting) |
| YOLOv8 segmentation | [`yolo`](https://github.com/newzjh/Aexis/releases/tag/yolo) |
| Real-ESRGAN | [`realesr`](https://github.com/newzjh/Aexis/releases/tag/realesr) |
| GFPGAN | [`gfpgan`](https://github.com/newzjh/Aexis/releases/tag/gfpgan) |
| DeepFillV2 | [`DeepFileV2`](https://github.com/newzjh/Aexis/releases/tag/DeepFileV2) |
| Stable Diffusion inpainting | [`model`](https://github.com/newzjh/Aexis/releases/tag/model), [`sdinpainting`](https://github.com/newzjh/Aexis/releases/tag/sdinpainting) |
| MONAI WholeBrain | [`MONAI_WholeBrain`](https://github.com/newzjh/Aexis/releases/tag/MONAI_WholeBrain) (external model and medical data only) |
| VISTA3D skull | [`vista3d_skull`](https://github.com/newzjh/Aexis/releases/tag/vista3d_skull) (external model and medical data only) |
| VISTA3D spine | [`vista3d_spine`](https://github.com/newzjh/Aexis/releases/tag/vista3d_spine) (external model and medical data only) |

Model weights have their own licenses and redistribution conditions. The listed release pages are download locations, not license grants.

## Install

For an embedded package, add the following entry to the consuming project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "individual.jiahua-zhang.aexis": "file:../path-to-aexis/Packages/individual.jiahua-zhang.aexis"
  }
}
```

For a registry or Git release, install the published `individual.jiahua-zhang.aexis` package through Unity Package Manager. Do not copy `Runtime` files into `Assets`; package-local resources and assembly definitions are part of the runtime contract.

## Package layout

| Path | Purpose |
| --- | --- |
| `Runtime/Core` | Backend-neutral contracts, tensor descriptors, precision and quantization manifests |
| `Runtime/Onnx` | Self-owned ONNX protobuf reader and execution-planning adapters |
| `Runtime/Ncnn` | Self-owned NCNN param/bin readers, graph loader, Pack4 texture execution, operators |
| `Runtime/Execution` | ONNX shape/index texture backend |
| `Runtime/Resources/Aexis` | Compute shaders loaded through `Resources`, owned by the package |
| `Editor` | Package editor tooling |
| `Samples~` | Unity-managed application example and model configuration; no model weights |
| `Documentation~` | Package documentation rendered by Package Manager |
| `Tests/Editor` | Package boundary and planning tests |

The runtime is partitioned into multiple asmdefs inside one UPM package. Unity supports this arrangement; consumers import only `individual.jiahua-zhang.aexis` and get the assemblies transitively. `Aexis.Ncnn` production inference stays on Pack4 RenderTextures and CommandBuffer-compatible texture flows. Compute buffers are limited to immutable uploads and explicit diagnostic paths.

`ForwardPack4WithFixedInputs(...)` records the same strict Pack4 CommandBuffer path for mixed texture/fixed-buffer model inputs. A fixed buffer is dispatched into a GPU texture before the first layer (including exact RFloat token-id upload for `Embed`); it cannot become an activation or trigger a CPU/ComputeBuffer fallback.

## Model import and extension contracts

Unity imports `.onnx`, NCNN `.param`, and versioned `.aexis` archive files as `AexisModelAsset`. The importer preserves source bytes, emits a versioned binary graph, attaches a sibling NCNN `.bin` when present, and records lowering/preflight diagnostics on the asset. `AexisModelPackager` exposes the same offline prepack path for build tooling.

`AexisNcnnBinaryParam` is the stable binary `.param` representation and `AexisModelArchive` (`AEXM` v1) packages the binary graph, weights, source, manifest, and diagnostics. Use `AexisCustomLayerRegistry` to register a public layer factory with a versioned parameter/arity schema. Model archives may declare their required custom layer type and shader kernel id; unresolved declarations fail before execution.

P1 visual operators (`GridSample`, deformable/ROI/detection families, `Fold`, `Flip`, `GLU`, `Einsum`, `Diag`, and `SPP`) ship with built-in Pack4 RenderTexture and CommandBuffer profiles. Strict preflight reuses the same parameter/shape proof as dispatch; an unsupported profile is rejected before execution rather than materialized through a ComputeBuffer. BF16 uses FP32 texture storage because Unity has no portable BF16 render-texture format; Pack4 `Cast` provides deterministic BF16 rounding. Per-layer mixed plans select FP16/FP32/BF16 physical storage as appropriate, while calibrated signed or unsigned Pack4 INT8 plans feed Conv/DWConv/Gemm/InnerProduct dispatch directly.

## Operator and quantization status

`output/operator-capabilities/operator-capabilities.json` in the repository is the implementation snapshot used for release documentation. It lists 96 NCNN import entries (56 with both RenderTexture and CommandBuffer flags) and 19 Sentis/ONNX import entries (13 with both flags). The snapshot contains 56 `partial`, 29 `debug-only`, 5 `alias-only`, and 6 `unsupported` entries, so a flag is not a blanket model compatibility promise. Aexis imports ONNX/Sentis-dialect graphs but does not depend on Sentis or ONNX Runtime at runtime.

`ModelManifest` can represent FP32, FP16, BF16, INT8/INT4 weight-only, calibrated W8A8, and per-layer mixed plans. Checked-in model manifests include model-specific INT8/INT4 profiles for MobileCLIP and Matting. The capability snapshot has zero universal per-operator FP16/INT8 flags; use strict preflight and the model's precision gate for a concrete graph. Qwen mobile Q4/Q8 are model archive variants, not a claim of generic engine-wide INT4/INT8 support.

## Tested environments

| Platform | Hardware and Unity | Current evidence | Status |
| --- | --- | --- | --- |
| Windows 11 Pro 64-bit | Intel Arc Graphics, Unity 6000.2.7f2, Vulkan | 2026-07-29: CodeFormer 17,958 ms; GFPGAN 6,053 ms; DeepFillV2 2,196 ms; Real-ESRGAN 1,057 ms; SD inpainting 630,213 ms | Passed for those runs |
| macOS Editor | Mac16,10, Apple M4, 16 GB, Unity 6000.2.7f2, Metal | 2026-07-30, 600 x 337 input: CLIP 11,755 ms; CodeFormer 15,332 ms; GFPGAN 4,696 ms; Real-ESRGAN 1,295 ms; Matting 2,182 ms; YOLO 986 ms (two people, 78.69% mask); YOLO + DeepFillV2 13,811 ms. Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. | Completed editor report; not a Player benchmark |
| macOS Player | Mac16,10, Apple M4, 16 GB, Unity 6000.2.7f2, Metal | 2026-07-30, 600 x 337 input: CLIP 97 ms; CodeFormer 2,515 ms; GFPGAN 1,136 ms; Real-ESRGAN 707 ms; Matting 421 ms; YOLO 294 ms (two people, 78.69% mask); YOLO + DeepFillV2 3,528 ms. Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. | Completed Player report |
| Android (MuMu emulator) | MuMu ADB `127.0.0.1:16384`; runtime reports vivo V2241A, Adreno (TM) 650, Unity 6000.2.7f2, Vulkan GameActivity | 2026-07-26 full Main2 pass: CodeFormer 6,946 ms; Real-ESRGAN 519 ms; GFPGAN 3,160 ms; YOLO 657 ms (four persons); DeepFillV2 3,442 ms; Matting 1,014 ms; CLIP 2,208 ms; Qwen Q4 326,821 ms (103 steps, 397 visible characters) | Passed; conservative Android fallback |
| iPadOS Simulator (`iPad8,6` profile) | Apple M4 host GPU, 16 GB host memory, Unity 6000.2.7f2, Metal | 2026-07-30, 600 x 337 input: CLIP 111 ms; CodeFormer 2,974 ms; GFPGAN 1,336 ms; Real-ESRGAN 903 ms; Matting 486 ms; YOLO 475 ms (two people, 78.69% mask); YOLO + DeepFillV2 4,731 ms. Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. | Completed simulator report; not a physical-device benchmark |
| iPhone / iPad physical device | **Pending command-run report: device/SoC/GPU/Unity; Metal required** | Run the [iOS Metal device procedure](Documentation~/apple-runtime-smoke.md#ios-metal-device) and return the build JSON plus `runner-report=` console log. | Ready for target-machine device validation |

Validation must use a real graphics device. `-nographics` is not valid for Aexis shader, package, or runner validation.

MuMu is a conservative Android fallback measurement, not a physical-device benchmark. The iPadOS report is a simulator measurement, not a physical-device benchmark. A Snapdragon 888 handset has been reported as materially faster; its exact result remains TBD until device, build, thermal state, graphics API, runner configuration, and timing are recorded.

### Development runner report

MainView2 exposes a **Test** button only in the Unity Editor and Development Players. It sits immediately before the Chinese and English language buttons and uses the current history image to run CLIP, CodeFormer, GFPGAN, Real-ESRGAN, Matting, Qwen3.5, YOLO segmentation, YOLO + DeepFillV2, and YOLO + SD inpainting. Each runner receives a 600-second cancellation token. Missing local or bundled model groups are skipped and written to the report; the test never requests a model download.

One JSON report is updated after each runner at `Application.persistentDataPath/AexisDevelopmentRunnerTest_*.json`. Windows uses the native Shell to open the folder and select the report, macOS reveals the file in Finder, Android launches a compatible JSON-view intent, and iOS opens the native document preview. The report includes source-image metadata, platform/device information, runner status, elapsed time, output dimensions, person count, mask coverage, and error or timeout detail.

## Quick start

```csharp
using Aexis.Ncnn;

var ops = new NcnnOps();
var session = NcnnInferenceSessionFactory.Create(ops);
using var model = new NcnnBinReader(binStream);
await session.LoadModelAsync(paramText, model);
// Prepare the Pack4 input expected by the model, then execute ForwardPack4.
session.Release();
```

See [API manual](Documentation~/api-manual.md) for input/output ownership and [runner samples](Documentation~/runner-samples.md) for an importable component.

## Application example and model files

Import **AIImage Main2 Application Example** from Package Manager, then run `Tools/Aexis/Examples/Install Main2 Application StreamingAssets`. This is the single full example: it combines `Main2`, MainView2, DesignView, LibraryView, UI/application code, all runners including GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN, plus editor tests and debug tools.

Its installer copies the sample inference configuration to `Assets/StreamingAssets`, the Unity Player-included location. The runner catalog uses `Clip/...`, `CodeFormer/...`, `DeepFileV2/...`, `Matting/...`, `RealESRGAN/...`, and `Yolo/...` paths below that root. No model weights or tokenizer vocabularies are included. Supply models under their own applicable licenses; see [model distribution](Documentation~/model-distribution.md).

Reduced Main2 Player builds do not stage, move, or modify `Assets/StreamingAssets` or package sample `StreamingAssets`. `Tools/Aexis/Release/Build Reduced` selects locally installed model files after Unity has generated the Player output (or during Android Gradle asset generation). The Asset Store package itself contains no model files.

The complete sample retains its Editor NUnit sources. A default sample import excludes them through `AEXIS_INCLUDE_EDITOR_TESTS`; install Unity Test Framework and add that define symbol when running the included test suite.

<a id="license"></a>

## License and package boundaries

The Asset Store edition is governed by the Unity Asset Store End User License Agreement. There is no separate Aexis license in this UPM package. Aexis does not include Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, MONAI, or VISTA source/binaries as runtime dependencies. Compatibility targets do not imply affiliation or use of upstream code.

The application sample includes modified source copies of Json.NET 13.0.2 and UniTask under their respective MIT licenses. Their locations, modifications, and complete license texts are in [Third-Party Notices.txt](Third-Party%20Notices.txt). The Json.NET source revision and archive checksum are also recorded under `Samples~/AIImageApplicationExample/ThirdParty/AexisSampleJson`; no Newtonsoft or UniTask package is installed into `Assets`.

The sample uses Unity-supported BCL compression streams for PNG metadata and MONAI volume I/O; it does not ship a third-party compression implementation.

Models are separate artifacts with their own licenses and redistribution conditions. This package does not include model weights, tokenizer vocabularies, sample photographs, or documentation screenshots. Do not place those assets in a Store upload until the provenance and redistribution review in `Documentation~/model-distribution.md` is complete.

## Community links

- [GitHub repository](https://github.com/newzjh/Aexis)
- [Unity Asset Store](https://assetstore.unity.com/preview/397636/1435586)
- LinkedIn: coming soon
- Discord: coming soon

## Documentation

- [Installation and architecture](Documentation~/architecture.md)
- [Public API manual](Documentation~/api-manual.md)
- [Runner sample guide](Documentation~/runner-samples.md)
- [Model distribution and exclusions](Documentation~/model-distribution.md)
- [Changelog](CHANGELOG.md)
- [Third-party licenses](Third-Party%20Notices.txt)
