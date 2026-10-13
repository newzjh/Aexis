# Aexis UPM Publishing

## Package identity

This Asset Store product ships one UPM package:

| Field | Required value |
| --- | --- |
| Publisher namespace | `individual.jiahua-zhang` |
| Product namespace | `aexis` |
| Package technical name | `individual.jiahua-zhang.aexis` |
| Package source directory | `Packages/individual.jiahua-zhang.aexis` |
| `package.json` `name` | `individual.jiahua-zhang.aexis` |
| Minimum Unity editor | `2022.3.9f1` (`unity`: `2022.3`, `unityRelease`: `9f1`) |

The product must be configured as a single-package UPM product in Publisher Portal. Do not upload a `.unitypackage`, create an `Assets` bootstrap, or nest the UPM package below `Assets`.

## Release preparation

1. Confirm the Publisher Portal product draft technical name is `individual.jiahua-zhang.aexis`.
2. Keep the package directory and `package.json` `name` identical to that technical name. Keep importable content in `Samples~`, never `Samples` or `Assets`. Run `Tools/Aexis/Validation/Validate UPM Publishing Layout` in Unity to verify the local package identity and minimum Unity declaration.
3. Keep the package below the Asset Store UPM size limit of 550 MB. The package must not include unapproved model weights, private data, caches, or generated build output.
4. Complete the source, shader, sample, and model provenance audit in `Packages/individual.jiahua-zhang.aexis/LICENSE.md` and `Packages/individual.jiahua-zhang.aexis/Documentation~/model-distribution.md`.
5. Before the first Store upload, set the approved release version in `package.json`, update `CHANGELOG.md`, and retain that exact package content for the immutable version record.

## Validate and upload

Use Unity's Asset Store Publishing Tools from Unity 6000.2.7f2:

1. Open **Tools > Asset Store > Validator**.
2. Select **UPM** as the validation type, set the product category to **Tools**, run the Validator, and resolve every reported failure before upload.
3. Open **Tools > Asset Store > Uploader**.
4. Select the **UPM Packages** tab and upload `individual.jiahua-zhang.aexis`.
5. Follow the link returned by the uploader to finish product metadata, pricing, media, and submission in Publisher Portal.

The uploader matches the package technical name against the Publisher Portal draft. A mismatch in the namespace, package directory, or `package.json` name must be corrected before upload.

## Post-upload checks

- Verify the uploaded package reports the intended Unity compatibility range: 2022.3 through 6000.3.
- Install the uploaded UPM product through Package Manager in an isolated project, then import the Main2 sample and run its installer.
- Do not amend a version already uploaded to the UPM registry. Publish a new version for every package-content change.

## Sources

- [Unity UPM publishing workflow](https://docs.unity.com/en-us/asset-store/publishing/upm-packages/publish.md)
- [Create a UPM product draft](https://docs.unity.com/en-us/asset-store/publishing/upm-packages/create-draft.md)
- [Validate and upload a UPM package](https://docs.unity.com/en-us/asset-store/publishing/upm-packages/validate.md)
