---
name: aiimage-package-release
description: Maintain, restructure, document, or validate the AIImage Aexis Unity package release. Use when editing Packages/individual.jiahua-zhang.aexis, package samples, package asmdefs, release documentation, model distribution, or empty-project import validation across Unity 2022.3 through Unity 6000.3.
---

# Aexis Package Release

Follow this skill for every Aexis package release change.

## Package contract

- Publish one UPM package only: `Packages/individual.jiahua-zhang.aexis`.
- The Asset Store publisher namespace is `individual.jiahua-zhang`, the product namespace is `aexis`, and the immutable package technical name is `individual.jiahua-zhang.aexis`. Keep the package directory, `package.json` `name`, project dependency, lock-file entry, and Publisher Portal draft technical name identical.
- Never export or submit this package as a `.unitypackage`, place it below `Assets`, or add an `Assets` bootstrap. Those layouts create an improper hybrid UPM package.
- Keep engine code in `Aexis*` namespaces. `AIImage` is an application/example namespace and must not appear in Runtime.
- Keep multiple Runtime asmdefs only for real boundaries: `Aexis`, `Aexis.Async`, `Aexis.Onnx`, `Aexis.Ncnn`, and `Aexis.Execution`. Editor and tests have their own asmdefs.
- Default release editor: Unity `6000.2.7f2`, executable `C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe`. The current project remains on this version for normal development and default release validation.
- Supported editor range: Unity `2022.3` through Unity `6000.3` (Unity 6.3). The compatibility matrix uses `2022.3.9f1`, `2023.2.20f1c1`, `6000.2.7f2`, and `6000.3.2f1`.
- Never change the current project's Unity version to run a non-default compatibility check. Unity versions other than `6000.2.7f2` must be tested in a separately created empty project or a copy of the project in another directory, with its own `ProjectSettings/ProjectVersion.txt`, `Library`, `Temp`, and generated project files. Do not open, upgrade, or rewrite the current project's version for compatibility testing.
- The package manifest minimum is Unity `2022.3`; do not raise it to the current editor unless the public support range is intentionally changed.
- Keep `Aexis.Async` based only on BCL `Task`. Do not add UniTask, Sentis, ONNX Runtime, ncnn, MNN, native inference plugins, or `AIImage` dependencies to Runtime.

## Runtime and resources

- Put engine compute assets under `Runtime/Resources/Aexis`. Load them through package-owned `Resources` names.
- Never use `AssetDatabase`, a hard-coded `Packages/...` path, or an application `Assets/...` path from Runtime.
- Preserve Pack4 RenderTexture and texture-backed execution in NCNN production paths. Do not introduce a ComputeBuffer fallback except clearly isolated debug/inspection code.
- Record logical and physical storage shapes separately when adding tensor paths.

## Samples and models

- Put importable examples in `Samples`; never place application scenes, UI, private goldens, medical data, caches, or model tools in Runtime.
- Use the `AIImageApplicationExample` installer to copy player configuration and permitted model assets to `Assets/StreamingAssets`; runtime sample code must remain cross-platform and AssetDatabase-free.
- Carry default models only for Clip, CodeFormer, DeepFillV2, Matting, RealESRGAN, and YOLO after a provenance/license audit.
- Exclude GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN model files from package samples.
- Before adding any model, record source URL, immutable revision/checksum, license, copyright, conversion history, and redistribution approval.

## Required checks

1. Scan `Packages/individual.jiahua-zhang.aexis/Runtime` for forbidden dependencies and paths.
2. Run `dotnet build AIImage.sln -v minimal -m:1`.
3. Compile and perform the default release smoke test with Unity `6000.2.7f2` in the current project. Do not change the current project's editor version for this check.
4. When a release claim covers the full range, validate `2022.3.9f1`, `2023.2.20f1c1`, and `6000.3.2f1` only in separate empty projects or copied projects located outside the current project directory. Each isolated validation project must use its own `ProjectVersion.txt`, `Library`, `Temp`, and generated project files; it must not modify the current project.
5. In every isolated validation project, install the package through a `file:` dependency, compile with only the package, then import AIImage Main2 Application Example through Package Manager and run its installer. Do not export or import a `.unitypackage` as part of UPM release validation.
6. Before upload, run `Aexis/Validation/Validate UPM Publishing Layout`, then run the official Asset Store UPM Validator. Upload only through the official **UPM Packages** workflow; do not use the legacy Asset Store Tools **Local UPM Package** workflow because it produces a hybrid `.unitypackage` submission.
7. Update `README.md`, `Documentation~`, `Documents/Aexis-Engine-Integration-Manual.md`, and `Documents/AssetStore/UPM-Publishing.md` when public APIs, package identity, package layout, sample paths, model payload, or the Store workflow changes.

## Graphics-backed validation

- Release validation must initialize a real Unity graphics device because Aexis ships and executes compute shaders. Use `-batchmode -quit` with the selected editor and project path; `-nographics` is forbidden for package import, shader compilation, sample compilation, runner checks, and release smoke tests.
- `-nographics` may not be used as a shortcut when an editor or CI host lacks a graphics device. Move the test to a graphics-capable host and record the unavailable environment instead of claiming a passing validation.
- A valid batch invocation has the form:

  ```text
  "C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe" -batchmode -quit -projectPath "<validation-project>" -logFile "<log-file>"
  ```

- Review logs for C# errors, shader errors, import failures, and package-lock conflicts. HLSL performance warnings may be recorded separately, but they do not replace the required real-device import and compile check.

Read [release layout reference](references/release-layout.md) before moving package content or changing the validation flow.
