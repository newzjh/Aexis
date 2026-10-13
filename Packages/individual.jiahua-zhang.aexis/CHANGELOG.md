# Changelog

## 0.1.0-pre.1

- Consolidated the AIImage inference packages into the single `individual.jiahua-zhang.aexis` package.
- Adopted the verified UPM publisher namespace and matching package directory for Asset Store UPM publishing.
- Removed the hybrid `.unitypackage` exporter; Asset Store releases use the UPM Validator and UPM Packages uploader flow.
- Moved the importable application example under the UPM-standard `Samples~` directory and declared Unity `2022.3.9f1` as the minimum release.
- Introduced Aexis Core, ONNX, NCNN, execution, async, editor, and test assemblies.
- Moved inference compute shader assets into package-owned Resources paths.
