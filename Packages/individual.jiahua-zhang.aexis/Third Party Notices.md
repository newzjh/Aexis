# Third-Party Audit

The `Runtime` implementation is self-developed. Compatibility targets such as ncnn, Sentis, ONNX Runtime, ONNX, MNN, MONAI, and VISTA are not runtime dependencies and their source, binaries, models, and data are not included by Aexis Runtime.

Before an MIT release, record every copied or generated artifact with its upstream URL, immutable version, license, copyright notice, modification history, and redistribution approval. Verify compute shader provenance independently. Keep model checkpoints, medical data, private golden data, and application-specific tooling outside this package.

The package has no third-party Unity package dependencies. Unity 6000.2 or later supplies the built-in async primitive used by `Aexis.Async`.

`Samples~/AIImageApplicationExample/ThirdParty` contains the sample-only, namespace-isolated `AexisSampleAsync` source derived from UniTask. It compiles as `Aexis.Sample.Async`, never as the original assembly or namespace. This isolation prevents project import conflicts; it does not change the upstream licensing of those files. Preserve the source headers and complete the source URL, immutable version, license text, copyright, modification, and redistribution review before publishing an archive containing the sample. PNG metadata and medical-volume compression use BCL streams supplied by Unity's supported .NET profile; no SharpZipLib source or binary is included.
