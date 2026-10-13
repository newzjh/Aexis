# Aexis Unity Asset Store Submission Kit

This folder contains the English-language copy, capture plan, and upload-ready image set for an Aexis Asset Store submission.

## Contents

- [Technical details](Technical-Details.md): publisher-portal copy grounded in the package manifest and current validation evidence.
- [UPM publishing](UPM-Publishing.md): technical-name contract and the official validation, upload, and approval workflow.
- [Screenshots and videos](Screenshots-And-Videos.md): gallery order, runner provenance, video script, and capture checklist.
- [Marketing images](Marketing-Images.md): upload slots, exact image dimensions, and provenance.
- `MarketingImages/`: generated PNGs ready for Publisher Portal upload after visual review.

## Release Gate

This kit makes the listing reviewable; it does not remove the repository's release gates. Do not submit the package until all of the following are complete:

1. Complete the source, shader, sample, and model provenance audit required by `Packages/individual.jiahua-zhang.aexis/LICENSE.md` and `Packages/individual.jiahua-zhang.aexis/Documentation~/model-distribution.md`.
2. Keep only model assets with verified redistribution permission in the submitted package.
3. Confirm the Publisher Portal single-package product technical name is `individual.jiahua-zhang.aexis`, matching both the package directory and `package.json`.
4. Replace the pre-release package version `0.1.0-pre.1` with the approved release version before the first immutable Store upload.
5. Record physical iOS/iPadOS Metal evidence before advertising physical iOS support.
6. Run the package release checks on a real graphics device, including the required Unity smoke test and the Asset Store UPM Validator.

All claims in this kit distinguish verified runner evidence from product capability. Timings are dated, environment-specific observations, not performance guarantees.
