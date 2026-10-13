# Aexis Release Layout

```
Packages/individual.jiahua-zhang.aexis/
  Runtime/{Core,Async,Onnx,Ncnn,Execution,Resources/Aexis}
  Editor/
  Tests/Editor/
  Samples~/AIImageApplicationExample/{Scenes,Runtime,ReferenceRunners,Editor,StreamingAssets,ThirdParty}
  Documentation~/
```

`individual.jiahua-zhang.aexis` is the Asset Store technical name and must match the package directory, `package.json` `name`, project dependency, lock-file entry, and Publisher Portal draft. `AIImageApplicationExample` is the single full application sample. It contains Main2, application/UI code, every runner, Editor tooling, isolated sample dependencies, and the only permitted model payloads. Default release validation uses Unity `6000.2.7f2` in the current project. Compatibility validation for any other supported editor, including `2022.3` through Unity `6000.3` (Unity 6.3), must use a separately created empty project or a copy in another directory; never change or upgrade the current project's Unity version. Each isolated project owns its `ProjectVersion.txt`, `Library`, `Temp`, and generated `.csproj` files. Import the local package using a `file:` dependency, import the sample through Package Manager before validating its code, and run the sample installer before testing `StreamingAssets` paths. Validate and upload through the official Asset Store UPM workflow only; never export the package as a `.unitypackage` or put it below `Assets`. Do not make Unity-generated `Library`, `Temp`, project `.csproj`, model caches, generated golden results, or local binaries release artifacts.
