# Changelog

## Unreleased

- Removed package-level MIT licensing declarations for the Asset Store UPM edition and added `Third-Party Notices.txt` for modified sample dependencies.
- Removed unreviewed model files, tokenizer vocabularies, sample photographs, and documentation screenshots from the UPM package; the application sample now ships inference configuration only.
- Added UPM layout validation for independent-license declarations, the required notices file, and model payloads in sample StreamingAssets.
- Corrected sample model downloads to use the published `newzjh/Aexis` release tags and exact asset names, including dynamic Qwen3.5 Q4 weight shards.
- Marked model groups without a complete published asset set unavailable in the sample download tools.

## 0.1.0-pre.1

- Consolidated the AIImage inference packages into the single `individual.jiahua-zhang.aexis` package.
- Adopted the verified UPM publisher namespace and matching package directory for Asset Store UPM publishing.
- Removed the hybrid `.unitypackage` exporter; Asset Store releases use the UPM Validator and UPM Packages uploader flow.
- Moved the importable application example under the UPM-standard `Samples~` directory and declared Unity `2022.3.9f1` as the minimum release.
- Introduced Aexis Core, ONNX, NCNN, execution, async, editor, and test assemblies.
- Moved inference compute shader assets into package-owned Resources paths.
