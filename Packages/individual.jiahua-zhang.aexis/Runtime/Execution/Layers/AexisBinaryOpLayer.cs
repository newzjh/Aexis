using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisBinaryOpLayer : AexisBaseLayer
    {
        // [TTS-DIAG] tail binaryop 输入 staging（debug）。
        public static RenderTexture s_tailBinRt;
        public static RenderTexture s_tailBinRtB;
        public static string s_tailBinTag;
        // [TTS-DIAG] tail binaryop 输出 staging（FP16 断点定位）。
        public static RenderTexture s_tailOutRt;
        public static string s_tailOutTag;

        public AexisBinaryOpLayer() : base(AexisLayerTypes.BinaryOp, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            // [TTS-DIAG] 入口日志：只对尾部层
            if (layer.name.Contains("/dec/Add_") || layer.name.Contains("/dec/Div_") || layer.name.Contains("/dec/LeakyRelu_5"))
            {
                bool __canRt = CanExecuteRenderTexturePath(owner, layer, context);
                if (AexisGraphSession.s_ttsVerbose) UnityEngine.Debug.Log($"[TTS-DIAG] BINOP-ENTRY {layer.name} canRt={__canRt} bottoms=[{string.Join(",", layer.bottomNames)}]");
            }
            if (CanExecuteRenderTexturePath(owner, layer, context))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            // Prefer the concrete Pack4 implementation in strict production mode.
            // CanExecuteRenderTexturePath is a fast preflight and may reject a valid
            // low-dimensional scalar/broadcast representation conservatively.
            if (owner.ShouldBlockPack4BufferFallback())
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var opType = layer.GetInt(0, 0);
            var withScalar = layer.GetInt(1, 0);
            var scalarB = layer.GetFloat(2, 0f);

            if (withScalar != 0)
            {
                var aBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var aView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (aBuf == null)
                    throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

                var outBuf = owner.RentTempBuffer(aBuf.count, sizeof(float));
                owner.Ops.BinaryOpScalarBuf(aBuf, scalarB, aBuf.count, opType, outBuf);
                bufferBlobs[layer.topNames[0]] = outBuf;
                if (aView != null)
                    bufferViews[layer.topNames[0]] = new AexisTensorBuffer(outBuf, aView.dims, aView.w, aView.h, aView.d, aView.c, false);
                tempOwned.Add(outBuf);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                ? AexisGraphSession.CodeFormerSftAddLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
            var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                ? AexisGraphSession.CodeFormerSftMulLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
            var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;

            using var aReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var aBufFallback = aReadable?.buffer;
            var aViewFallback = aReadable;
            if (aBufFallback == null || aViewFallback == null)
                throw new InvalidOperationException("BinaryOp source not found: " + layer.name);

            using var bReadable = owner.GetReadableTensorInput(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var bBufFallback = bReadable?.buffer;
            var bViewFallback = bReadable;
            if (bBufFallback == null || bViewFallback == null)
                throw new InvalidOperationException("BinaryOp second source not found: " + layer.name);

            // [TTS-DIAG] buffer-path tail binary op 输入回读
            if (aBufFallback != null && aBufFallback.count >= 1000000)
            {
                try
                {
                    var __fa = new float[aBufFallback.count];
                    aBufFallback.GetData(__fa, 0, 0, aBufFallback.count);
                    float __mxa = 0; foreach (var v in __fa) { if (System.Math.Abs(v) > __mxa) __mxa = (float)System.Math.Abs(v); }
                    float __mxb = 0;
                    if (bBufFallback != null && bBufFallback.count >= 1000000)
                    {
                        var __fb = new float[bBufFallback.count];
                        bBufFallback.GetData(__fb, 0, 0, bBufFallback.count);
                        foreach (var v in __fb) { if (System.Math.Abs(v) > __mxb) __mxb = (float)System.Math.Abs(v); }
                    }
                    UnityEngine.Debug.Log($"[TTS-DIAG] BUFPATH BINOP {layer.name} op={opType} aCount={aBufFallback.count} aMax={__mxa:F4} bMax={__mxb:F4} a0={layer.bottomNames[0]} b1={layer.bottomNames[1]}");
                }
                catch (System.Exception __bx) { UnityEngine.Debug.Log($"[TTS-DIAG] BUFPATH readback fail {layer.name}: {__bx.Message}"); }
            }

            if (aViewFallback.dims == 2 && bViewFallback.dims == 2 && aBufFallback.count != bBufFallback.count)
            {
                if (owner.TryExpand2DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
                else if (owner.TryExpand2DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }
            else if (aViewFallback.dims == 1 && bViewFallback.dims == 2)
            {
                if (owner.TryExpand1DTo2DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }
            else if (aViewFallback.dims == 2 && bViewFallback.dims == 1)
            {
                if (owner.TryExpand1DTo2DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
            }
            else if (aViewFallback.dims == 3 && bViewFallback.dims == 3)
            {
                if (owner.TryExpand3DBroadcastBuffer(bBufFallback, bViewFallback, aViewFallback, out var expandedB, out var expandedBView))
                {
                    bBufFallback = expandedB;
                    bViewFallback = expandedBView;
                    tempOwned.Add(expandedB);
                }
                else if (owner.TryExpand3DBroadcastBuffer(aBufFallback, aViewFallback, bViewFallback, out var expandedA, out var expandedAView))
                {
                    aBufFallback = expandedA;
                    aViewFallback = expandedAView;
                    tempOwned.Add(expandedA);
                }
            }

            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
            {
                var scaledB = owner.RentTempBuffer(bBufFallback.count, sizeof(float));
                owner.Ops.CopyBuf(bBufFallback, scaledB, bBufFallback.count);
                owner.Ops.MulScalarInplace(scaledB, owner.CodeFormerSftAddScale, scaledB.count);
                bBufFallback = scaledB;
                tempOwned.Add(scaledB);
            }

            var broadcast = AexisGraphSession.ResolveBinaryBroadcast(aViewFallback, bViewFallback, aBufFallback.count, bBufFallback.count, layer.name);
            var outBufFallback = owner.RentTempBuffer(broadcast.total, sizeof(float));
            owner.Ops.BinaryOpBuf(aBufFallback, bBufFallback, broadcast.total, opType, outBufFallback, broadcast.mode, broadcast.size);
            if (isCodeFormerSftMul)
            {
                if (owner.CodeFormerBypassSftMul)
                {
                    owner.Ops.CopyBuf(aBufFallback, outBufFallback, broadcast.total);
                }
                else if (owner.CodeFormerSftMulScale != 1f)
                {
                    owner.Ops.MulScalarInplace(outBufFallback, owner.CodeFormerSftMulScale, outBufFallback.count);
                }
            }

            var outTensorFallback = broadcast.outputView != null
                ? new AexisTensorBuffer(outBufFallback, broadcast.outputView.dims, broadcast.outputView.w, broadcast.outputView.h, broadcast.outputView.d, broadcast.outputView.c, true, owner.ReturnTempBuffer)
                : new AexisTensorBuffer(outBufFallback, 1, outBufFallback.count, 1, 1, 1, true, owner.ReturnTempBuffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensorFallback,
                preferTexture: broadcast.outputView != null && broadcast.outputView.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var opType = layer.GetInt(0, 0);
            var withScalar = layer.GetInt(1, 0);
            var scalarB = layer.GetFloat(2, 0f);
            var scalarTextureFormat = AexisGraphSession.ResolveTensorTextureFormat(2);

            // [TTS-DIAG] tail render-texture binary-op 输入探针（verbose 开关）
            if (AexisGraphSession.s_ttsVerbose
                && layer.bottomNames.Length >= 1
                && AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var __aT, out var __aS)
                && __aT.texture != null && __aT.texture.width >= 16000)
            {
                var __sb = new System.Text.StringBuilder();
                __sb.Append($"[TTS-DIAG] RT-PATH BINOP {layer.name} op={opType} withScalar={withScalar} a0={layer.bottomNames[0]}(w={__aT.texture.width},h={__aT.texture.height})");
                for (int bi = 1; bi < layer.bottomNames.Length; bi++)
                {
                    if (AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[bi], out var __bT, out var __bS))
                        __sb.Append($" b{bi}={layer.bottomNames[bi]}(w={__bT.texture.width})");
                    else
                        __sb.Append($" b{bi}={layer.bottomNames[bi]}(NOTEX)");
                }
                UnityEngine.Debug.Log(__sb.ToString());
                // 回读两个输入 slice0
                try
                {
                    var __gta = new Texture2D(__aT.texture.width, __aT.texture.height, TextureFormat.RGBAFloat, false, true);
                    Graphics.SetRenderTarget(__aT.texture, 0);
                    __gta.ReadPixels(new Rect(0, 0, __aT.texture.width, __aT.texture.height), 0, 0);
                    __gta.Apply();
                    float __mxa = 0; foreach (var p in __gta.GetPixels()) __mxa = Mathf.Max(__mxa, Mathf.Abs(p.r) + Mathf.Abs(p.g) + Mathf.Abs(p.b) + Mathf.Abs(p.a));
                    UnityEngine.Debug.Log($"[TTS-DIAG] RT-PATH BINOP a0 readback max={__mxa:F4}");
                    UnityEngine.Object.DestroyImmediate(__gta);
                    for (int bi = 1; bi < layer.bottomNames.Length; bi++)
                    {
                        if (AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[bi], out var __bt2, out _) && __bt2.texture != null)
                        {
                            var __gtb = new Texture2D(__bt2.texture.width, __bt2.texture.height, TextureFormat.RGBAFloat, false, true);
                            Graphics.SetRenderTarget(__bt2.texture, 0);
                            __gtb.ReadPixels(new Rect(0, 0, __bt2.texture.width, __bt2.texture.height), 0, 0);
                            __gtb.Apply();
                            float __mxb = 0; foreach (var p in __gtb.GetPixels()) __mxb = Mathf.Max(__mxb, Mathf.Abs(p.r) + Mathf.Abs(p.g) + Mathf.Abs(p.b) + Mathf.Abs(p.a));
                            UnityEngine.Debug.Log($"[TTS-DIAG] RT-PATH BINOP b{bi} readback max={__mxb:F4}");
                            UnityEngine.Object.DestroyImmediate(__gtb);
                        }
                    }
                }
                catch (System.Exception __dx) { UnityEngine.Debug.Log($"[TTS-DIAG] RT-PATH readback fail {layer.name}: {__dx.Message}"); }
            }

            if (withScalar != 0)
            {
                if (owner.ShouldForceCurrentLayerBufferPath()
                    || !AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var aTexScalar, out var aTexShapeScalar)
                    || !CanUseScalarLikeTexturePath(aTexScalar, aTexShapeScalar))
                {
                    throw new InvalidOperationException("BinaryOp render-texture scalar path requires existing texture input: " + layer.name);
                }

                var aTexStorageShapeScalar = AexisGraphSession.GetTextureStorageShape(aTexScalar, aTexShapeScalar);
                if (AexisGraphSession.IsPack4LinearMatTexture(aTexScalar, aTexShapeScalar))
                {
                    var storageShape = AexisGraphSession.GetTextureStorageShape(aTexScalar, aTexShapeScalar);
                    var outRt = owner.RentTempArray(storageShape.w, storageShape.h, 1, aTexScalar.texture.format);
                    owner.Ops.BinaryOpScalarPack4(aTexScalar.texture, scalarB, 1, opType, outRt);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, aTexShapeScalar, storageShape);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                if (AexisGraphSession.IsStrictLinearMatTexture(aTexScalar))
                {
                    var outLinear = owner.RentTempMat(aTexStorageShapeScalar.w, aTexStorageShapeScalar.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.BinaryOpScalarLinearMat(aTexScalar.texture, scalarB, opType, outLinear);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outLinear, aTexShapeScalar, aTexStorageShapeScalar);
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }

                RenderTexture aScalarMaterialized = null;
                var aScalarInput = MaterializeScalarLikeTexture(owner, aTexScalar, aTexShapeScalar, scalarTextureFormat, ref aScalarMaterialized);
                var scalarDepth = ResolveTexturePhysicalDepth(aScalarInput, aTexScalar.packs);
                var outRtScalar = owner.RentTempArray(
                    Mathf.Max(1, aScalarInput.width),
                    Mathf.Max(1, aScalarInput.height),
                    scalarDepth,
                    AexisGraphSession.ResolveTensorTextureFormat(aTexShapeScalar.dims));
                owner.Ops.BinaryOpScalarPack4(aScalarInput, scalarB, scalarDepth, opType, outRtScalar);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRtScalar, aTexShapeScalar, aTexStorageShapeScalar);
                if (aScalarMaterialized != null)
                    owner.ReturnTempArray(aScalarMaterialized);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                ? AexisGraphSession.CodeFormerSftAddLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
            var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                ? AexisGraphSession.CodeFormerSftMulLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
            var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;

            AexisGraphSession.TensorRef aTex = null;
            AexisGraphSession.TensorRef bTex = null;
            AexisGraphSession.BufferShape aTexShape = default;
            AexisGraphSession.BufferShape bTexShape = default;
            var canUseTextureBinary = false;
            var pack4ATex = aTex;
            var pack4BTex = bTex;
            var pack4AShape = aTexShape;
            var pack4BShape = bTexShape;

            if (TryResolveExactPack4LinearBinaryPath(
                    textureBlobs,
                    textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out var pack4LinearA,
                    out var pack4LinearAShape,
                    out var pack4LinearB,
                    out _,
                    out var pack4LinearStorageShape))
            {
                RenderTexture finalTexture = null;
                try
                {
                    finalTexture = owner.RentTempArray(pack4LinearStorageShape.w, pack4LinearStorageShape.h, 1, pack4LinearA.texture.format);
                    owner.Ops.BinaryOpPack4(pack4LinearA.texture, pack4LinearB.texture, 1, opType, finalTexture);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, pack4LinearAShape, pack4LinearStorageShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolvePack4LinearMixedBinaryPath(
                    textureBlobs,
                    textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out var mixedPack4Linear,
                    out var mixedLinear,
                    out var mixedOutShape,
                    out var mixedStorageShape,
                    out var mixedPack4IsA))
            {
                RenderTexture finalTexture = null;
                try
                {
                    finalTexture = owner.RentTempArray(
                        mixedStorageShape.w,
                        mixedStorageShape.h,
                        ResolveTexturePhysicalDepth(mixedPack4Linear.texture, mixedPack4Linear.packs),
                        mixedPack4Linear.texture.format);
                    if (mixedLinear.texture.dimension == TextureDimension.Tex2DArray)
                        owner.Ops.BinaryOpPack4LinearMixedArray(mixedPack4Linear.texture, mixedLinear.texture, mixedPack4IsA, opType, finalTexture);
                    else
                        owner.Ops.BinaryOpPack4LinearMixed(mixedPack4Linear.texture, mixedLinear.texture, mixedPack4IsA, opType, finalTexture);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, mixedOutShape, mixedStorageShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.DebugLog?.Invoke(
                    "[Texture][BinaryOpPack4LinearMixed]"
                    + " | layer=" + layer.name
                    + " | pack4IsA=" + (mixedPack4IsA ? "1" : "0")
                    + " | out=d" + mixedOutShape.dims + ":" + mixedOutShape.w + "x" + mixedOutShape.h + "x" + mixedOutShape.d + "x" + mixedOutShape.c
                    + " | storage=d" + mixedStorageShape.dims + ":" + mixedStorageShape.w + "x" + mixedStorageShape.h + "x" + mixedStorageShape.d + "x" + mixedStorageShape.c);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolveLinearMatScalarArrayBinaryPath(
                    textureBlobs,
                    textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out var scalarArrayLinear,
                    out var scalarArray,
                    out var scalarArrayShape,
                    out var scalarArrayStorageShape,
                    out var linearIsA))
            {
                RenderTexture finalTexture = null;
                try
                {
                    finalTexture = owner.RentTempMat(
                        scalarArrayStorageShape.w,
                        scalarArrayStorageShape.h,
                        AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.BinaryOpLinearMatScalarArray(
                        scalarArrayLinear.texture,
                        scalarArray.texture,
                        linearIsA,
                        opType,
                        finalTexture);
                    AexisGraphSession.SetTextureBlob(
                        textureBlobs,
                        textureShapes,
                        layer.topNames[0],
                        finalTexture,
                        scalarArrayShape,
                        scalarArrayStorageShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            canUseTextureBinary = !owner.ForceBufferBinaryOpAll
                && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out aTex, out aTexShape)
                && owner.TryGetPack4Texture(layer.bottomNames[1], textureBlobs, textureShapes, bufferBlobs, bufferViews, out bTex, out bTexShape);
            pack4ATex = aTex;
            pack4BTex = bTex;
            pack4AShape = aTexShape;
            pack4BShape = bTexShape;

            if (canUseTextureBinary && AexisGraphSession.CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
            {
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var rhsTexture = bTex.texture;
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        var scaledBDepth = bTexShape.dims == 4 ? bTexShape.d * bTex.packs : bTex.packs;
                        scaledBTexture = owner.RentTempArray(bTex.width, bTex.height, scaledBDepth, RenderTextureFormat.ARGBHalf);
                        owner.Ops.ScalePack4(bTex.texture, owner.CodeFormerSftAddScale, bTex.packs, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    var exactOutDepth = aTexShape.dims == 4 ? aTexShape.d * aTex.packs : aTex.packs;
                    finalTexture = owner.RentTempArray(
                        aTex.width,
                        aTex.height,
                        exactOutDepth,
                        owner.ResolveActivationTextureFormat(aTexShape.dims));
                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        owner.Ops.CopyPack4(aTex.texture, 0, finalTexture, 0, aTex.packs);
                    }
                    else
                    {
                        owner.Ops.BinaryOpPack4(aTex.texture, rhsTexture, aTex.packs, opType, finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(aTex.width, aTex.height, exactOutDepth, RenderTextureFormat.ARGBHalf);
                            owner.Ops.ScalePack4(finalTexture, owner.CodeFormerSftMulScale, aTex.packs, scaledOutTexture);
                            owner.ReturnTempArray(finalTexture);
                            finalTexture = scaledOutTexture;
                        }
                    }

                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, aTexShape);
                    finalTexture = null;
                }
                finally
                {
                    if (scaledBTexture != null)
                        owner.ReturnTempArray(scaledBTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolveExactScalar2DTextureBinaryPath(
                textureBlobs,
                textureShapes,
                layer.bottomNames[0],
                layer.bottomNames[1],
                out var scalar2DATex,
                out var scalar2DAShape,
                out var scalar2DBTex,
                out var scalar2DBShape))
            {
                RenderTexture aScalarMaterialized = null;
                RenderTexture bScalarMaterialized = null;
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    if (CanUseExactLinearMatBinaryPath(
                            scalar2DATex,
                            scalar2DAShape,
                            scalar2DBTex,
                            scalar2DBShape,
                            out var scalar2DStorageShape))
                    {
                        var rhsLinear = scalar2DBTex.texture;
                        if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                        {
                            scaledBTexture = owner.RentTempMat(scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                            owner.Ops.BinaryOpScalarLinearMat(rhsLinear, owner.CodeFormerSftAddScale, 2, scaledBTexture);
                            rhsLinear = scaledBTexture;
                        }

                        finalTexture = owner.RentTempMat(scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                        if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                        {
                            owner.Ops.BinaryOpScalarLinearMat(scalar2DATex.texture, 0f, 0, finalTexture);
                        }
                        else
                        {
                            owner.Ops.BinaryOpLinearMat(scalar2DATex.texture, rhsLinear, opType, finalTexture);
                            if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                            {
                                var scaledOutTexture = owner.RentTempMat(scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                owner.Ops.BinaryOpScalarLinearMat(finalTexture, owner.CodeFormerSftMulScale, 2, scaledOutTexture);
                                owner.ReturnTempArray(finalTexture);
                                finalTexture = scaledOutTexture;
                            }
                        }

                        AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalar2DAShape, scalar2DStorageShape);
                        finalTexture = null;
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    var lhsTexture = MaterializeScalarLikeTexture(owner, scalar2DATex, scalar2DAShape, scalarTextureFormat, ref aScalarMaterialized);
                    var rhsTexture = MaterializeScalarLikeTexture(owner, scalar2DBTex, scalar2DBShape, scalarTextureFormat, ref bScalarMaterialized);
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        scaledBTexture = owner.RentTempArray(scalar2DBTex.width, scalar2DBTex.height, 1, scalarTextureFormat);
                        owner.Ops.ScalePack4(rhsTexture, owner.CodeFormerSftAddScale, 1, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    finalTexture = owner.RentTempArray(scalar2DATex.width, scalar2DATex.height, 1, scalarTextureFormat);
                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        owner.Ops.CopyPack4(lhsTexture, 0, finalTexture, 0, 1);
                    }
                    else
                    {
                        owner.Ops.BinaryOpPack4(lhsTexture, rhsTexture, 1, opType, finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(scalar2DATex.width, scalar2DATex.height, 1, scalarTextureFormat);
                            owner.Ops.ScalePack4(finalTexture, owner.CodeFormerSftMulScale, 1, scaledOutTexture);
                            owner.ReturnTempArray(finalTexture);
                            finalTexture = scaledOutTexture;
                        }
                    }

                    var scalarPack4StorageShape = new AexisGraphSession.BufferShape(
                        3,
                        finalTexture.width,
                        finalTexture.height,
                        1,
                        1);
                    AexisGraphSession.SetTextureBlob(
                        textureBlobs,
                        textureShapes,
                        layer.topNames[0],
                        finalTexture,
                        scalar2DAShape,
                        scalarPack4StorageShape);
                    finalTexture = null;
                }
                finally
                {
                    ReturnTempUnique(owner, ref bScalarMaterialized, aScalarMaterialized, null);
                    if (aScalarMaterialized != null)
                        owner.ReturnTempArray(aScalarMaterialized);
                    if (scaledBTexture != null)
                        owner.ReturnTempArray(scaledBTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (TryResolveScalarSingleBroadcastTextureBinaryPath(
                    textureBlobs,
                    textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out var scalarBroadA,
                    out var scalarBroadAShape,
                    out var scalarBroadB,
                    out var scalarBroadBShape,
                    out var scalarBroadcastMode,
                    out var scalarBroadcastOutShape,
                    out var scalarBroadcastStorageShape))
            {
                RenderTexture scalarBroadAMaterialized = null;
                RenderTexture scalarBroadBMaterialized = null;
                RenderTexture scaledBTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var lhsTexture = MaterializeScalarLikeTexture(owner, scalarBroadA, scalarBroadAShape, scalarTextureFormat, ref scalarBroadAMaterialized);
                    var rhsTexture = MaterializeScalarLikeTexture(owner, scalarBroadB, scalarBroadBShape, scalarTextureFormat, ref scalarBroadBMaterialized);
                    if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                    {
                        scaledBTexture = owner.RentTempArray(scalarBroadB.width, scalarBroadB.height, 1, scalarTextureFormat);
                        owner.Ops.ScalePack4(rhsTexture, owner.CodeFormerSftAddScale, 1, scaledBTexture);
                        rhsTexture = scaledBTexture;
                    }

                    finalTexture = owner.RentTempArray(
                        scalarBroadcastStorageShape.w,
                        scalarBroadcastStorageShape.h,
                        1,
                        scalarTextureFormat);

                    if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                    {
                        var zeroTexture = owner.RentTempArray(rhsTexture.width, rhsTexture.height, 1, scalarTextureFormat);
                        owner.Ops.BinaryOpScalarPack4(rhsTexture, 0f, 1, 2, zeroTexture);
                        owner.Ops.BinaryOpScalarSingleBroadcast(
                            lhsTexture,
                            zeroTexture,
                            scalarBroadcastStorageShape.w,
                            scalarBroadcastStorageShape.h,
                            0,
                            scalarBroadcastMode,
                            finalTexture);
                        owner.ReturnTempArray(zeroTexture);
                    }
                    else
                    {
                        owner.Ops.BinaryOpScalarSingleBroadcast(
                            lhsTexture,
                            rhsTexture,
                            scalarBroadcastStorageShape.w,
                            scalarBroadcastStorageShape.h,
                            opType,
                            scalarBroadcastMode,
                            finalTexture);
                        if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                        {
                            var scaledOutTexture = owner.RentTempArray(finalTexture.width, finalTexture.height, 1, scalarTextureFormat);
                            owner.Ops.ScalePack4(finalTexture, owner.CodeFormerSftMulScale, 1, scaledOutTexture);
                            owner.ReturnTempArray(finalTexture);
                            finalTexture = scaledOutTexture;
                        }
                    }

                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarBroadcastOutShape, scalarBroadcastStorageShape);
                    finalTexture = null;
                }
                finally
                {
                    ReturnTempUnique(owner, ref scalarBroadBMaterialized, scalarBroadAMaterialized, null);
                    if (scalarBroadAMaterialized != null)
                        owner.ReturnTempArray(scalarBroadAMaterialized);
                    if (scaledBTexture != null)
                        owner.ReturnTempArray(scaledBTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            // A rank-1 MemoryData scalar is published as a 1x1 RFloat
            // Texture2D, while the other operand is a Pack4 Texture2DArray.
            // Keep this mixed pair on the immediate GPU path; binding the
            // scalar to the Pack4 array slot would otherwise force the legacy
            // buffer conversion path (and is rejected in strict Pack4 mode).
            if (TryResolvePack4TextureScalarTexture(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var packScalarTexture,
                    out var packScalarShape,
                    out var rhsScalarTexture,
                    out var rhsScalarShape))
            {
                RenderTexture finalTexture = null;
                try
                {
                    var outDepth = packScalarShape.dims == 4
                        ? Mathf.Max(1, packScalarShape.d * packScalarTexture.packs)
                        : Mathf.Max(1, packScalarTexture.packs);
                    finalTexture = owner.RentTempArray(
                        packScalarTexture.width,
                        packScalarTexture.height,
                        outDepth,
                        owner.ResolveActivationTextureFormat(packScalarShape.dims));
                    owner.Ops.BinaryOpPack4TexScalar(
                        packScalarTexture.texture,
                        rhsScalarTexture.texture,
                        packScalarTexture.packs,
                        opType,
                        finalTexture);
                    AexisGraphSession.SetTextureBlob(
                        textureBlobs,
                        textureShapes,
                        layer.topNames[0],
                        finalTexture,
                        packScalarShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4TextureChannelVectorBroadcast(
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    out var vectorTextureTex,
                    out var vectorTextureTexShape,
                    out var channelVectorTextureTex,
                    out var channelVectorTextureTexShape,
                    out var channelVectorTextureIsA))
            {
                RenderTexture channelVectorMaterialized = null;
                RenderTexture finalTexture = null;
                try
                {
                    var channelVectorInput = MaterializeScalarLikeTexture(owner, channelVectorTextureTex, channelVectorTextureTexShape, scalarTextureFormat, ref channelVectorMaterialized);
                    var vectorOutDepth = vectorTextureTexShape.dims == 4 ? vectorTextureTexShape.d * vectorTextureTex.packs : vectorTextureTex.packs;
                    finalTexture = owner.RentTempArray(vectorTextureTex.width, vectorTextureTex.height, vectorOutDepth, RenderTextureFormat.ARGBHalf);
                    owner.Ops.BinaryOpPack4ChannelVectorTex(
                        vectorTextureTex.texture,
                        channelVectorInput,
                        vectorTextureTex.packs,
                        opType,
                        channelVectorTextureIsA,
                        finalTexture);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, vectorTextureTexShape);
                    finalTexture = null;
                }
                finally
                {
                    if (channelVectorMaterialized != null)
                        owner.ReturnTempArray(channelVectorMaterialized);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4ChannelVectorBroadcast(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var vectorTexture,
                    out var vectorTextureShape,
                    out var channelVectorView,
                    out var channelVectorIsA))
            {
                RenderTexture vectorBroadcastTexture = null;
                RenderTexture finalTexture = null;
                try
                {
                    var packedChannelView = channelVectorView.Reshape(3, 1, 1, 1, vectorTextureShape.c);
                    vectorBroadcastTexture = owner.MaterializeTextureFromBufferView(channelVectorView.buffer, packedChannelView);
                    if (vectorBroadcastTexture == null)
                        throw new InvalidOperationException("Failed to materialize BinaryOp channel vector texture: " + layer.name);

                    var vectorOutDepth = vectorTextureShape.dims == 4 ? vectorTextureShape.d * vectorTexture.packs : vectorTexture.packs;
                    finalTexture = owner.RentTempArray(vectorTexture.width, vectorTexture.height, vectorOutDepth, RenderTextureFormat.ARGBHalf);
                    if (channelVectorIsA)
                        owner.Ops.BinaryOpPack4Broadcast(vectorBroadcastTexture, vectorTexture.texture, vectorTexture.packs, opType, 1, finalTexture);
                    else
                        owner.Ops.BinaryOpPack4Broadcast(vectorTexture.texture, vectorBroadcastTexture, vectorTexture.packs, opType, 2, finalTexture);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, vectorTextureShape);
                    finalTexture = null;
                }
                finally
                {
                    if (vectorBroadcastTexture != null)
                        owner.ReturnTempArray(vectorBroadcastTexture);
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4BufferScalar(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var scalarTexture,
                    out var scalarTextureShape,
                    out var scalarBuffer,
                    out var scalarIsA))
            {
                RenderTexture finalTexture = null;
                try
                {
                    var scalarStorageShape = AexisGraphSession.GetTextureStorageShape(scalarTexture, scalarTextureShape);
                    if (AexisGraphSession.IsStrictLinearMatTexture(scalarTexture))
                    {
                        finalTexture = owner.RentTempMat(scalarStorageShape.w, scalarStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                        owner.Ops.BinaryOpLinearMatFixedInputScalar(scalarTexture.texture, scalarBuffer, opType, scalarIsA, finalTexture);
                        AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarTextureShape, scalarStorageShape);
                    }
                    else
                    {
                        var scalarOutDepth = scalarTextureShape.dims == 4 ? scalarTextureShape.d * scalarTexture.packs : scalarTexture.packs;
                        finalTexture = owner.RentTempArray(scalarTexture.width, scalarTexture.height, scalarOutDepth, RenderTextureFormat.ARGBHalf);
                        owner.Ops.BinaryOpPack4BufferScalar(scalarTexture.texture, scalarBuffer, scalarTexture.packs, opType, scalarIsA, finalTexture);
                        AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, scalarTextureShape);
                    }
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            if (canUseTextureBinary
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4SpatialBroadcast(pack4ATex, pack4AShape, pack4BTex, pack4BShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
            {
                RenderTexture finalTexture = null;
                try
                {
                    var spatialOutDepth = outShape.dims == 4 ? outShape.d * outPacks : outPacks;
                    finalTexture = owner.RentTempArray(outWidth, outHeight, spatialOutDepth, RenderTextureFormat.ARGBHalf);
                    owner.Ops.BinaryOpPack4Broadcast(pack4ATex.texture, pack4BTex.texture, outPacks, opType, broadcastMode, finalTexture);
                    AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], finalTexture, outShape);
                    finalTexture = null;
                }
                finally
                {
                    if (finalTexture != null)
                        owner.ReturnTempArray(finalTexture);
                }

                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            owner.DebugLog?.Invoke(
                "[BinaryOpRtUnsupported]"
                + " | layer=" + (layer.name ?? string.Empty)
                + " | opType=" + opType.ToString(CultureInfo.InvariantCulture)
                + " | canUseTextureBinary=" + canUseTextureBinary.ToString()
                + " | exact=" + (canUseTextureBinary && AexisGraphSession.CanUseExactPack4BinaryPath(pack4ATex, pack4AShape, pack4BTex, pack4BShape)).ToString()
                + " | spatial=" + (canUseTextureBinary && TryResolvePack4SpatialBroadcast(pack4ATex, pack4AShape, pack4BTex, pack4BShape, out _, out _, out _, out _, out _)).ToString()
                + " | aTex=" + DescribeTensorRef(pack4ATex, pack4AShape)
                + " | bTex=" + DescribeTensorRef(pack4BTex, pack4BShape));
            throw new InvalidOperationException("BinaryOp render-texture path unsupported config: " + layer.name);
        }

        private static string DescribeTensorRef(AexisGraphSession.TensorRef tensor, AexisGraphSession.BufferShape shape)
        {
            if (tensor == null || tensor.texture == null)
                return "null";

            return tensor.width.ToString(CultureInfo.InvariantCulture)
                + "x" + tensor.height.ToString(CultureInfo.InvariantCulture)
                + "x" + tensor.packs.ToString(CultureInfo.InvariantCulture)
                + "p logical=d" + shape.dims.ToString(CultureInfo.InvariantCulture)
                + ":" + shape.w.ToString(CultureInfo.InvariantCulture)
                + "x" + shape.h.ToString(CultureInfo.InvariantCulture)
                + "x" + shape.d.ToString(CultureInfo.InvariantCulture)
                + "x" + shape.c.ToString(CultureInfo.InvariantCulture)
                + " storage=" + AexisGraphSession.GetTextureStorageShape(tensor, shape).ToString();
        }

        private static bool TryResolvePack4SpatialBroadcast(
            AexisGraphSession.TensorRef aTex,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.TensorRef bTex,
            AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (aTex == null || bTex == null || aTex.texture == null || bTex.texture == null)
                return false;
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;

            var matchingSpatialExtent = aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d;
            if (matchingSpatialExtent
                && aShape.c == 1
                && bShape.c > 1
                && aTex.packs == 1
                && bTex.packs == Mathf.Max(1, Mathf.CeilToInt(bShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(aTex, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(bTex, bShape))
            {
                broadcastMode = 3;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingSpatialExtent
                && bShape.c == 1
                && aShape.c > 1
                && bTex.packs == 1
                && aTex.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(aTex, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(bTex, bShape))
            {
                broadcastMode = 4;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            var matchingRowsAndChannels = aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && aTex.packs == bTex.packs
                && aTex.height == bTex.height
                && aTex.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(aTex, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(bTex, bShape);
            if (matchingRowsAndChannels
                && aShape.w == 1
                && bShape.w > 1
                && aTex.width == 1
                && bTex.width == bShape.w)
            {
                broadcastMode = 5;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingRowsAndChannels
                && bShape.w == 1
                && aShape.w > 1
                && bTex.width == 1
                && aTex.width == aShape.w)
            {
                broadcastMode = 6;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            // 深度轴广播：空间尺寸与 Pack4 通道打包完全一致，仅逻辑深度轴一侧为 1
            // （batch 共享的相对位置注意力项）。输出取存活一侧的深度，内核按 pack
            // 索引复用单层操作数。
            var matchingPackedSpatial = aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.c == bShape.c
                && aTex.packs == bTex.packs
                && aTex.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && aTex.width == bTex.width
                && aTex.height == bTex.height
                && AexisGraphSession.MatchesPack4TextureStorage(aTex, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(bTex, bShape);
            if (matchingPackedSpatial && aShape.d == 1 && bShape.d > 1)
            {
                broadcastMode = 11;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingPackedSpatial && bShape.d == 1 && aShape.d > 1)
            {
                broadcastMode = 10;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            if (TryResolvePack4WidthVectorBroadcast(
                    aTex,
                    aShape,
                    bTex,
                    bShape,
                    out broadcastMode,
                    out outShape,
                    out outWidth,
                    out outHeight,
                    out outPacks))
            {
                return true;
            }

            if (aShape.c != bShape.c || aTex.packs != bTex.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && aShape.d == 1 && aTex.width == 1 && aTex.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && bTex.width == 1 && bTex.height == 1;
            var aIsOutputSpatial = aShape.w == aTex.width && aShape.h == aTex.height;
            var bIsOutputSpatial = bShape.w == bTex.width && bShape.h == bTex.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4WidthVectorBroadcast(
            AexisGraphSession.TensorRef aTex,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.TensorRef bTex,
            AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (aTex == null || bTex == null || aTex.texture == null || bTex.texture == null)
                return false;
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;

            if (IsPack4WidthVector(aTex, aShape)
                && IsPack4WidthVectorTarget(bTex, bShape, aShape.w))
            {
                broadcastMode = 7;
                outWidth = bTex.width;
                outHeight = bTex.height;
                outPacks = bTex.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (IsPack4WidthVector(bTex, bShape)
                && IsPack4WidthVectorTarget(aTex, aShape, bShape.w))
            {
                broadcastMode = 8;
                outWidth = aTex.width;
                outHeight = aTex.height;
                outPacks = aTex.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            return false;
        }

        private static bool IsPack4WidthVector(AexisGraphSession.TensorRef texture, AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h == 1
                && shape.d == 1
                && shape.c == 1
                && texture.width == shape.w
                && texture.height == 1
                && texture.packs == 1
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool IsPack4WidthVectorTarget(AexisGraphSession.TensorRef texture, AexisGraphSession.BufferShape shape, int width)
        {
            return texture != null
                && texture.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w == width
                && shape.h > 0
                && shape.d > 0
                && shape.c > 0
                && texture.width == width
                && texture.packs == Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool TryResolvePack4ChannelVectorBroadcast(
            AexisGraphSession owner,
            string aName,
            string bName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape textureShape,
            out AexisTensorBuffer channelVector,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorIsA = false;

            if (owner == null)
                return false;

            if (TryGetChannelVectorBuffer(bName, bufferBlobs, bufferViews, out var bVector)
                && owner.TryGetPack4Texture(aName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aShape)
                && aShape.dims == 3
                && bVector.elementCount == aShape.c)
            {
                texture = aTex;
                textureShape = aShape;
                channelVector = bVector;
                channelVectorIsA = false;
                return true;
            }

            if (TryGetChannelVectorBuffer(aName, bufferBlobs, bufferViews, out var aVector)
                && owner.TryGetPack4Texture(bName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bShape)
                && bShape.dims == 3
                && aVector.elementCount == bShape.c)
            {
                texture = bTex;
                textureShape = bShape;
                channelVector = aVector;
                channelVectorIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4TextureChannelVectorBroadcast(
            string aName,
            string bName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape textureShape,
            out AexisGraphSession.TensorRef channelVector,
            out AexisGraphSession.BufferShape channelVectorShape,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorShape = default;
            channelVectorIsA = false;

            if (TryGetExistingChannelVectorTexture(textureBlobs, textureShapes, bName, out var bVector, out var bLogicalShape, out _)
                && AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, aName, out var aTex, out var aShape)
                && (aShape.dims == 3 || aShape.dims == 4)
                && AexisGraphSession.MatchesPack4TextureStorage(aTex, aShape)
                && MatchesChannelVectorWidth(bLogicalShape, aShape.c))
            {
                texture = aTex;
                textureShape = aShape;
                channelVector = bVector;
                channelVectorShape = bLogicalShape;
                channelVectorIsA = false;
                return true;
            }

            if (TryGetExistingChannelVectorTexture(textureBlobs, textureShapes, aName, out var aVector, out var aLogicalShape, out _)
                && AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, bName, out var bTex, out var bShape)
                && (bShape.dims == 3 || bShape.dims == 4)
                && AexisGraphSession.MatchesPack4TextureStorage(bTex, bShape)
                && MatchesChannelVectorWidth(aLogicalShape, bShape.c))
            {
                texture = bTex;
                textureShape = bShape;
                channelVector = aVector;
                channelVectorShape = aLogicalShape;
                channelVectorIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4BufferScalar(
            AexisGraphSession owner,
            string aName,
            string bName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape textureShape,
            out ComputeBuffer scalar,
            out bool scalarIsA)
        {
            texture = null;
            textureShape = default;
            scalar = null;
            scalarIsA = false;

            if (owner == null)
                return false;

            if (TryGetSingleElementBuffer(bName, bufferBlobs, bufferViews, out var bScalar)
                && owner.TryGetPack4Texture(aName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var aTex, out var aShape))
            {
                texture = aTex;
                textureShape = aShape;
                scalar = bScalar;
                scalarIsA = false;
                return true;
            }

            if (TryGetSingleElementBuffer(aName, bufferBlobs, bufferViews, out var aScalar)
                && owner.TryGetPack4Texture(bName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var bTex, out var bShape))
            {
                texture = bTex;
                textureShape = bShape;
                scalar = aScalar;
                scalarIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4TextureScalarTexture(
            AexisGraphSession owner,
            string aName,
            string bName,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.TensorRef packTexture,
            out AexisGraphSession.BufferShape packShape,
            out AexisGraphSession.TensorRef scalarTexture,
            out AexisGraphSession.BufferShape scalarShape)
        {
            packTexture = null;
            packShape = default;
            scalarTexture = null;
            scalarShape = default;
            if (owner == null)
                return false;

            // The dedicated immediate kernel implements tensor / scalar. Keep
            // the scalar on the RHS so subtraction and division preserve ONNX
            // operand order; scalar-on-LHS uses the general broadcast path.
            if (!owner.TryGetPack4Texture(
                    aName,
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferViews,
                    out var aPack,
                    out var aPackShape)
                || !AexisGraphSession.TryGetExistingTexture(
                    textureBlobs,
                    textureShapes,
                    bName,
                    out var bScalar,
                    out var bScalarShape)
                || !IsImmediateScalarTexture(bScalar, bScalarShape))
            {
                return false;
            }

            packTexture = aPack;
            packShape = aPackShape;
            scalarTexture = bScalar;
            scalarShape = bScalarShape;
            return true;
        }

        private static bool IsImmediateScalarTexture(
            AexisGraphSession.TensorRef texture,
            AexisGraphSession.BufferShape shape)
        {
            if (texture == null || texture.texture == null
                || texture.texture.dimension != TextureDimension.Tex2D)
                return false;
            if (shape.dims == 1)
                return shape.w == 1 && shape.h <= 1 && shape.d <= 1 && shape.c <= 1;
            if (shape.dims == 2)
                return shape.w == 1 && shape.h == 1 && shape.c <= 1;
            return shape.dims >= 3
                && shape.w == 1 && shape.h == 1 && shape.d == 1 && shape.c == 1;
        }

        private static bool TryResolveExactScalar2DTextureBinaryPath(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string aName,
            string bName,
            out AexisGraphSession.TensorRef aTex,
            out AexisGraphSession.BufferShape aShape,
            out AexisGraphSession.TensorRef bTex,
            out AexisGraphSession.BufferShape bShape)
        {
            aTex = null;
            aShape = default;
            bTex = null;
            bShape = default;

            if (!TryGetScalar2DTexture(textureBlobs, textureShapes, aName, out aTex, out aShape)
                || !TryGetScalar2DTexture(textureBlobs, textureShapes, bName, out bTex, out bShape))
            {
                return false;
            }

            return aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && aTex.width == bTex.width
                && aTex.height == bTex.height
                && aTex.packs == bTex.packs;
        }

        private static bool TryResolveExactPack4LinearBinaryPath(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string aName,
            string bName,
            out AexisGraphSession.TensorRef aTex,
            out AexisGraphSession.BufferShape aShape,
            out AexisGraphSession.TensorRef bTex,
            out AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            aTex = null;
            aShape = default;
            bTex = null;
            bShape = default;
            storageShape = default;

            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, aName, out aTex, out var aContract)
                || !AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, bName, out bTex, out var bContract))
                return false;

            aShape = aContract.LogicalShape;
            bShape = bContract.LogicalShape;
            if (aShape.w != bShape.w
                || aShape.h != bShape.h
                || aShape.d != bShape.d
                || aShape.c != bShape.c
                || !AexisGraphSession.IsPack4LinearMatTexture(aTex, aShape)
                || !AexisGraphSession.IsPack4LinearMatTexture(bTex, bShape))
                return false;

            var aStorage = aContract.StorageShape;
            var bStorage = bContract.StorageShape;
            if (!AexisGraphSession.BufferShapeEquals(aStorage, bStorage))
                return false;

            storageShape = aStorage;
            return true;
        }

        private static bool TryResolvePack4LinearMixedBinaryPath(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string aName,
            string bName,
            out AexisGraphSession.TensorRef pack4Linear,
            out AexisGraphSession.TensorRef linear,
            out AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape storageShape,
            out bool pack4IsA)
        {
            pack4Linear = null;
            linear = null;
            logicalShape = default;
            storageShape = default;
            pack4IsA = false;

            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, aName, out var aTex, out var aContract)
                || !AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, bName, out var bTex, out var bContract))
                return false;

            var aShape = aContract.LogicalShape;
            var bShape = bContract.LogicalShape;
            if (aShape.dims != 2
                || bShape.dims != 2
                || aShape.w != bShape.w
                || aShape.h != bShape.h
                || aShape.d != bShape.d
                || aShape.c != bShape.c)
            {
                return false;
            }

            var aPack4 = IsPackedLogical2DTexture(aTex, aShape);
            var bPack4 = IsPackedLogical2DTexture(bTex, bShape);
            var aLinear = AexisGraphSession.IsStrictLinearMatTexture(aTex)
                || IsScalarLinearArrayTexture(aTex, aShape, aContract.StorageShape);
            var bLinear = AexisGraphSession.IsStrictLinearMatTexture(bTex)
                || IsScalarLinearArrayTexture(bTex, bShape, bContract.StorageShape);
            if (aPack4 == bPack4)
                return false;

            if (aPack4 && bLinear)
            {
                pack4Linear = aTex;
                linear = bTex;
                logicalShape = aShape;
                storageShape = aContract.StorageShape;
                pack4IsA = true;
                return true;
            }

            if (bPack4 && aLinear)
            {
                pack4Linear = bTex;
                linear = aTex;
                logicalShape = bShape;
                storageShape = bContract.StorageShape;
                pack4IsA = false;
                return true;
            }

            return false;
        }

        private static bool TryResolveLinearMatScalarArrayBinaryPath(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string aName,
            string bName,
            out AexisGraphSession.TensorRef linear,
            out AexisGraphSession.TensorRef scalarArray,
            out AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape storageShape,
            out bool linearIsA)
        {
            linear = null;
            scalarArray = null;
            logicalShape = default;
            storageShape = default;
            linearIsA = false;

            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, aName, out var aTex, out var aContract)
                || !AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, bName, out var bTex, out var bContract)
                || !AexisGraphSession.BufferShapeEquals(aContract.LogicalShape, bContract.LogicalShape))
            {
                return false;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(aTex)
                && IsScalarLinearArrayTexture(bTex, bContract.LogicalShape, bContract.StorageShape))
            {
                linear = aTex;
                scalarArray = bTex;
                logicalShape = aContract.LogicalShape;
                storageShape = aContract.StorageShape;
                linearIsA = true;
                return true;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(bTex)
                && IsScalarLinearArrayTexture(aTex, aContract.LogicalShape, aContract.StorageShape))
            {
                linear = bTex;
                scalarArray = aTex;
                logicalShape = bContract.LogicalShape;
                storageShape = bContract.StorageShape;
                linearIsA = false;
                return true;
            }

            return false;
        }

        private static bool TryResolvePack4LinearMixedBinaryPath(
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            string aName,
            string bName,
            out AexisGraphSession.CmdTensorRef pack4Linear,
            out AexisGraphSession.CmdTensorRef linear,
            out AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape storageShape,
            out bool pack4IsA)
        {
            pack4Linear = null;
            linear = null;
            logicalShape = default;
            storageShape = default;
            pack4IsA = false;

            if (!AexisGraphSession.TryGetExistingCmdTextureContract(blobs, shapes, aName, out var aTex, out var aContract)
                || !AexisGraphSession.TryGetExistingCmdTextureContract(blobs, shapes, bName, out var bTex, out var bContract))
                return false;

            var aShape = aContract.LogicalShape;
            var bShape = bContract.LogicalShape;
            if (aShape.dims != 2
                || bShape.dims != 2
                || aShape.w != bShape.w
                || aShape.h != bShape.h
                || aShape.d != bShape.d
                || aShape.c != bShape.c)
            {
                return false;
            }

            var aPack4 = IsPackedLogical2DTexture(aTex, aShape);
            var bPack4 = IsPackedLogical2DTexture(bTex, bShape);
            var aLinear = AexisGraphSession.IsStrictLinearMatTexture(aTex)
                || IsScalarLinearArrayTexture(aTex, aShape, aContract.StorageShape);
            var bLinear = AexisGraphSession.IsStrictLinearMatTexture(bTex)
                || IsScalarLinearArrayTexture(bTex, bShape, bContract.StorageShape);
            if (aPack4 == bPack4)
                return false;

            if (aPack4 && bLinear)
            {
                pack4Linear = aTex;
                linear = bTex;
                logicalShape = aShape;
                storageShape = aContract.StorageShape;
                pack4IsA = true;
                return true;
            }

            if (bPack4 && aLinear)
            {
                pack4Linear = bTex;
                linear = aTex;
                logicalShape = bShape;
                storageShape = bContract.StorageShape;
                pack4IsA = false;
                return true;
            }

            return false;
        }

        private static bool TryResolveLinearMatScalarArrayBinaryPath(
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            string aName,
            string bName,
            out AexisGraphSession.CmdTensorRef linear,
            out AexisGraphSession.CmdTensorRef scalarArray,
            out AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape storageShape,
            out bool linearIsA)
        {
            linear = null;
            scalarArray = null;
            logicalShape = default;
            storageShape = default;
            linearIsA = false;

            if (!AexisGraphSession.TryGetExistingCmdTextureContract(blobs, shapes, aName, out var aTex, out var aContract)
                || !AexisGraphSession.TryGetExistingCmdTextureContract(blobs, shapes, bName, out var bTex, out var bContract)
                || !AexisGraphSession.BufferShapeEquals(aContract.LogicalShape, bContract.LogicalShape))
            {
                return false;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(aTex)
                && IsScalarLinearArrayTexture(bTex, bContract.LogicalShape, bContract.StorageShape))
            {
                linear = aTex;
                scalarArray = bTex;
                logicalShape = aContract.LogicalShape;
                storageShape = aContract.StorageShape;
                linearIsA = true;
                return true;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(bTex)
                && IsScalarLinearArrayTexture(aTex, aContract.LogicalShape, aContract.StorageShape))
            {
                linear = bTex;
                scalarArray = aTex;
                logicalShape = bContract.LogicalShape;
                storageShape = bContract.StorageShape;
                linearIsA = false;
                return true;
            }

            return false;
        }

        private static bool IsScalarLinearArrayTexture(
            AexisGraphSession.TensorRef tensor,
            AexisGraphSession.BufferShape logicalShape,
            AexisGraphSession.BufferShape storageShape)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.texture.dimension == TextureDimension.Tex2DArray
                && tensor.layoutKind == AexisTextureTensorLayoutKind.LinearMat
                && logicalShape.dims == 2
                && storageShape.dims == 3
                && storageShape.w == logicalShape.w
                && storageShape.h == logicalShape.h
                && storageShape.d == 1
                && storageShape.c == 1
                && tensor.width == storageShape.w
                && tensor.height == storageShape.h
                && tensor.packs == 1
                && ResolveTexturePhysicalDepth(tensor.texture, tensor.packs) == 1;
        }

        private static bool IsScalarLinearArrayTexture(
            AexisGraphSession.CmdTensorRef tensor,
            AexisGraphSession.BufferShape logicalShape,
            AexisGraphSession.BufferShape storageShape)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.texture.dimension == TextureDimension.Tex2DArray
                && tensor.layoutKind == AexisTextureTensorLayoutKind.LinearMat
                && logicalShape.dims == 2
                && storageShape.dims == 3
                && storageShape.w == logicalShape.w
                && storageShape.h == logicalShape.h
                && storageShape.d == 1
                && storageShape.c == 1
                && tensor.width == storageShape.w
                && tensor.height == storageShape.h
                && tensor.packs == 1
                && ResolveTexturePhysicalDepth(tensor.texture, tensor.packs) == 1;
        }

        private static bool IsPackedLogical2DTexture(AexisGraphSession.TensorRef tensor, AexisGraphSession.BufferShape shape)
        {
            if (AexisGraphSession.IsPack4LinearMatTexture(tensor, shape))
                return true;
            if (tensor == null
                || tensor.texture == null
                || shape.dims != 2
                || tensor.texture.dimension != TextureDimension.Tex2DArray
                || tensor.height != shape.h)
            {
                return false;
            }
            var slices = ResolveTexturePhysicalDepth(tensor.texture, tensor.packs);
            if (slices <= 1 || tensor.width >= shape.w)
                return false;
            var logicalPacks = checked((shape.w + 3) / 4);
            return checked(tensor.width * slices) >= logicalPacks
                && checked(tensor.width * (slices - 1)) < logicalPacks;
        }

        private static bool IsPackedLogical2DTexture(AexisGraphSession.CmdTensorRef tensor, AexisGraphSession.BufferShape shape)
        {
            if (AexisGraphSession.IsPack4LinearMatTexture(tensor, shape))
                return true;
            if (tensor == null
                || tensor.texture == null
                || shape.dims != 2
                || tensor.height != shape.h)
            {
                return false;
            }
            var slices = ResolveTexturePhysicalDepth(tensor.texture, tensor.packs);
            if (slices <= 1 || tensor.width >= shape.w)
                return false;
            var logicalPacks = checked((shape.w + 3) / 4);
            return checked(tensor.width * slices) >= logicalPacks
                && checked(tensor.width * (slices - 1)) < logicalPacks;
        }

        private static bool TryResolveScalarSingleBroadcastTextureBinaryPath(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string aName,
            string bName,
            out AexisGraphSession.TensorRef aTex,
            out AexisGraphSession.BufferShape aShape,
            out AexisGraphSession.TensorRef bTex,
            out AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            aTex = null;
            aShape = default;
            bTex = null;
            bShape = default;
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            if (!TryGetScalarLikeTexture(textureBlobs, textureShapes, aName, out aTex, out aShape)
                || !TryGetScalarLikeTexture(textureBlobs, textureShapes, bName, out bTex, out bShape))
            {
                return false;
            }

            return TryResolveScalarSingleBroadcastShapes(aShape, bShape, out broadcastMode, out outShape, out storageShape);
        }

        private static bool TryGetScalar2DTexture(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string name,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape shape)
        {
            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, name, out texture, out var contract))
            {
                shape = default;
                return false;
            }
            shape = contract.LogicalShape;
            return CanUseScalar2DTexturePath(texture, shape);
        }

        private static bool TryGetScalarLikeTexture(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string name,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, name, out texture, out var contract))
                return false;
            shape = contract.LogicalShape;
            return CanUseScalarLikeTexturePath(texture, shape);
        }

        private static bool TryGetExistingChannelVectorTexture(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string name,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            texture = null;
            logicalShape = default;
            storageShape = default;
            if (!AexisGraphSession.TryGetExistingTextureContract(textureBlobs, textureShapes, name, out texture, out var contract))
                return false;
            logicalShape = contract.LogicalShape;
            storageShape = contract.StorageShape;
            return CanUseChannelVectorTexturePath(texture, logicalShape, storageShape);
        }

        private static int ResolveTexturePhysicalDepth(RenderTexture texture, int fallbackDepth)
        {
            if (texture == null)
                return Mathf.Max(1, fallbackDepth);
            return Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : fallbackDepth);
        }

        private static int ResolveTexturePhysicalDepth(ComputeTexture texture, int fallbackDepth)
        {
            if (texture == null)
                return Mathf.Max(1, fallbackDepth);
            return Mathf.Max(1, texture.depth > 0 ? texture.depth : fallbackDepth);
        }

        private static bool CanUseScalar2DTexturePath(AexisGraphSession.TensorRef texture, AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static bool CanUseExactLinearMatBinaryPath(
            AexisGraphSession.TensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.TensorRef b,
            AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            storageShape = default;
            if (!CanUseScalar2DTexturePath(a, aShape)
                || !CanUseScalar2DTexturePath(b, bShape)
                || !AexisGraphSession.IsStrictLinearMatTexture(a)
                || !AexisGraphSession.IsStrictLinearMatTexture(b))
                return false;
            if (aShape.w != bShape.w
                || aShape.h != bShape.h
                || aShape.d != bShape.d
                || aShape.c != bShape.c)
                return false;

            var aStorage = AexisGraphSession.GetTextureStorageShape(a, aShape);
            var bStorage = AexisGraphSession.GetTextureStorageShape(b, bShape);
            if (aStorage.dims != bStorage.dims
                || aStorage.w != bStorage.w
                || aStorage.h != bStorage.h
                || a.texture.width != aStorage.w
                || a.texture.height != aStorage.h
                || b.texture.width != bStorage.w
                || b.texture.height != bStorage.h)
                return false;

            storageShape = aStorage;
            return true;
        }

        private static bool CanUseScalarLikeTexturePath(AexisGraphSession.TensorRef texture, AexisGraphSession.BufferShape shape)
        {
            if (texture == null || texture.texture == null)
                return false;
            if (AexisGraphSession.IsPack4LinearMatTexture(texture, shape))
                return true;
            if (texture.packs == 1
                && ((shape.dims == 1 && shape.w > 0 && texture.width == shape.w && texture.height == 1)
                    || (shape.dims == 2 && shape.w > 0 && shape.h > 0 && texture.width == shape.w && texture.height == shape.h)))
            {
                return true;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(texture))
                return false;

            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h > 0
                && shape.c > 0
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool CanUseChannelVectorTexturePath(AexisGraphSession.TensorRef texture, AexisGraphSession.BufferShape logicalShape, AexisGraphSession.BufferShape storageShape)
        {
            if (texture == null || texture.texture == null || texture.packs != 1)
                return false;
            if (logicalShape.dims == 3
                && logicalShape.w == 1 && logicalShape.h > 1
                && logicalShape.d == 1 && logicalShape.c == 1)
            {
                return storageShape.w == 1
                    && storageShape.h == logicalShape.h
                    && texture.width == 1
                    && texture.height == logicalShape.h;
            }
            if ((logicalShape.dims != 1 && logicalShape.dims != 2) || logicalShape.w <= 0)
                return false;
            if (storageShape.w == logicalShape.w && storageShape.h == 1 && texture.width == logicalShape.w && texture.height == 1)
                return true;
            if (storageShape.w == 1 && storageShape.h == logicalShape.w && texture.width == 1 && texture.height == logicalShape.w)
                return true;
            return false;
        }

        private static bool CanUseScalar2DTexturePath(AexisGraphSession.CmdTensorRef texture, AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static bool CanUseExactLinearMatBinaryPath(
            AexisGraphSession.CmdTensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef b,
            AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            storageShape = default;
            if (!CanUseScalar2DTexturePath(a, aShape)
                || !CanUseScalar2DTexturePath(b, bShape)
                || !AexisGraphSession.IsStrictLinearMatTexture(a)
                || !AexisGraphSession.IsStrictLinearMatTexture(b))
                return false;
            if (aShape.w != bShape.w
                || aShape.h != bShape.h
                || aShape.d != bShape.d
                || aShape.c != bShape.c)
                return false;

            var aStorage = AexisGraphSession.GetCmdStorageShape(a, aShape);
            var bStorage = AexisGraphSession.GetCmdStorageShape(b, bShape);
            if (aStorage.dims != bStorage.dims
                || aStorage.w != bStorage.w
                || aStorage.h != bStorage.h
                || a.width != aStorage.w
                || a.height != aStorage.h
                || b.width != bStorage.w
                || b.height != bStorage.h)
                return false;

            storageShape = aStorage;
            return true;
        }

        private static bool CanUseScalarLikeTexturePath(AexisGraphSession.CmdTensorRef texture, AexisGraphSession.BufferShape shape)
        {
            if (texture == null || texture.texture == null)
                return false;
            if (AexisGraphSession.IsPack4LinearMatTexture(texture, shape))
                return true;
            if (texture.packs == 1
                && ((shape.dims == 1 && shape.w > 0 && texture.width == shape.w && texture.height == 1)
                    || (shape.dims == 2 && shape.w > 0 && shape.h > 0 && texture.width == shape.w && texture.height == shape.h)))
            {
                return true;
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(texture))
                return false;

            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h > 0
                && shape.c > 0
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static RenderTexture MaterializeScalarLikeTexture(
            AexisGraphSession owner,
            AexisGraphSession.TensorRef source,
            AexisGraphSession.BufferShape shape,
            RenderTextureFormat format,
            ref RenderTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
                return source.texture;

            var outH = shape.dims == 1 ? 1 : Mathf.Max(1, shape.h);
            var outDims = shape.dims == 1 ? 1 : 2;
            materialized = owner.RentTempArray(Mathf.Max(1, shape.w), outH, 1, format);
            owner.Ops.ReshapeLinearMatToPack4(
                source.texture,
                shape.w,
                outH,
                shape.w,
                outH,
                1,
                1,
                outDims,
                materialized);
            return materialized;
        }

        private static ComputeTexture MaterializeScalarLikeTexture(
            AexisGraphSession owner,
            CommandBuffer cmd,
            AexisGraphSession.CmdTensorRef source,
            AexisGraphSession.BufferShape shape,
            RenderTextureFormat format,
            ref ComputeTexture materialized,
            string plannedResourceName = null)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
                return source.texture;

            var outH = shape.dims == 1 ? 1 : Mathf.Max(1, shape.h);
            var outDims = shape.dims == 1 ? 1 : 2;
            materialized = string.IsNullOrWhiteSpace(plannedResourceName)
                ? owner.RentTempArray(cmd, Mathf.Max(1, shape.w), outH, 1, format)
                : owner.RentTempArray(cmd, Mathf.Max(1, shape.w), outH, 1, format, plannedResourceName);
            owner.Ops.ReshapeLinearMatToPack4(
                cmd,
                source.texture,
                shape.w,
                outH,
                shape.w,
                outH,
                1,
                1,
                outDims,
                materialized);
            return materialized;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, ref RenderTexture texture, RenderTexture alias0, RenderTexture alias1)
        {
            if (texture == null || ReferenceEquals(texture, alias0) || ReferenceEquals(texture, alias1))
            {
                texture = null;
                return;
            }

            owner.ReturnTempArray(texture);
            texture = null;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, CommandBuffer cmd, ref ComputeTexture texture, ComputeTexture alias0, ComputeTexture alias1)
        {
            if (texture == null)
                return;
            if ((alias0 != null && texture.nameID == alias0.nameID) || (alias1 != null && texture.nameID == alias1.nameID))
            {
                texture = null;
                return;
            }

            owner.ReturnTempArray(cmd, texture);
            texture = null;
        }

        private static bool CanUseChannelVectorTexturePath(AexisGraphSession.CmdTensorRef texture, AexisGraphSession.BufferShape logicalShape, AexisGraphSession.BufferShape storageShape)
        {
            if (texture == null || texture.texture == null || texture.packs != 1)
                return false;
            if (logicalShape.dims == 3
                && logicalShape.w == 1 && logicalShape.h > 1
                && logicalShape.d == 1 && logicalShape.c == 1)
            {
                return storageShape.w == 1
                    && storageShape.h == logicalShape.h
                    && texture.width == 1
                    && texture.height == logicalShape.h;
            }
            if ((logicalShape.dims != 1 && logicalShape.dims != 2) || logicalShape.w <= 0)
                return false;
            if (storageShape.w == logicalShape.w && storageShape.h == 1 && texture.width == logicalShape.w && texture.height == 1)
                return true;
            if (storageShape.w == 1 && storageShape.h == logicalShape.w && texture.width == 1 && texture.height == logicalShape.w)
                return true;
            return false;
        }

        private static bool MatchesChannelVectorWidth(AexisGraphSession.BufferShape logicalShape, int channels)
        {
            return (logicalShape.w == channels
                    && (logicalShape.dims == 1 || (logicalShape.dims == 2 && logicalShape.h == 1)))
                || (logicalShape.dims == 3
                    && logicalShape.w == 1
                    && logicalShape.h == channels
                    && logicalShape.d == 1
                    && logicalShape.c == 1);
        }

        private static bool TryResolveScalarSingleBroadcastCmdBinaryPath(
            AexisGraphSession.CmdTensorRef aTex,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef bTex,
            AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.CmdTensorRef outATex,
            out AexisGraphSession.CmdTensorRef outBTex,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            outATex = null;
            outBTex = null;
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            if (!CanUseScalarLikeTexturePath(aTex, aShape) || !CanUseScalarLikeTexturePath(bTex, bShape))
                return false;
            if (!TryResolveScalarSingleBroadcastShapes(aShape, bShape, out broadcastMode, out outShape, out storageShape))
                return false;

            outATex = aTex;
            outBTex = bTex;
            return true;
        }

        private static bool TryResolveCmdPack4TextureChannelVectorBroadcast(
            AexisGraphSession.CmdTensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef b,
            AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.CmdTensorRef texture,
            out AexisGraphSession.BufferShape textureShape,
            out AexisGraphSession.CmdTensorRef channelVector,
            out AexisGraphSession.BufferShape channelVectorShape,
            out bool channelVectorIsA)
        {
            texture = null;
            textureShape = default;
            channelVector = null;
            channelVectorShape = default;
            channelVectorIsA = false;

            if ((aShape.dims == 3 || aShape.dims == 4)
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && CanUseChannelVectorTexturePath(b, bShape, AexisGraphSession.GetCmdStorageShape(b, bShape))
                && MatchesChannelVectorWidth(bShape, aShape.c))
            {
                texture = a;
                textureShape = aShape;
                channelVector = b;
                channelVectorShape = bShape;
                channelVectorIsA = false;
                return true;
            }

            if ((bShape.dims == 3 || bShape.dims == 4)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape)
                && CanUseChannelVectorTexturePath(a, aShape, AexisGraphSession.GetCmdStorageShape(a, aShape))
                && MatchesChannelVectorWidth(aShape, bShape.c))
            {
                texture = b;
                textureShape = bShape;
                channelVector = a;
                channelVectorShape = aShape;
                channelVectorIsA = true;
                return true;
            }

            return false;
        }

        private static bool TryResolveCmdPack4ChannelLayoutReshape(
            AexisGraphSession.CmdTensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef b,
            AexisGraphSession.BufferShape bShape,
            out AexisGraphSession.CmdTensorRef target,
            out AexisGraphSession.BufferShape targetShape,
            out AexisGraphSession.CmdTensorRef source,
            out AexisGraphSession.BufferShape sourceShape,
            out bool sourceIsA)
        {
            target = null;
            targetShape = default;
            source = null;
            sourceShape = default;
            sourceIsA = false;

            if (IsPack4ChannelLayoutTarget(a, aShape)
                && IsPack4ChannelLayoutSource(b, bShape, aShape))
            {
                target = a;
                targetShape = aShape;
                source = b;
                sourceShape = bShape;
                return true;
            }

            if (IsPack4ChannelLayoutTarget(b, bShape)
                && IsPack4ChannelLayoutSource(a, aShape, bShape))
            {
                target = b;
                targetShape = bShape;
                source = a;
                sourceShape = aShape;
                sourceIsA = true;
                return true;
            }

            return false;
        }

        private static bool IsPack4ChannelLayoutTarget(
            AexisGraphSession.CmdTensorRef tensor,
            AexisGraphSession.BufferShape shape)
        {
            return tensor != null
                && tensor.texture != null
                && shape.dims == 3
                && shape.w > 0
                && shape.h == 1
                && shape.d == 1
                && shape.c > 1
                && tensor.packs == Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(tensor, shape);
        }

        private static bool IsPack4ChannelLayoutSource(
            AexisGraphSession.CmdTensorRef tensor,
            AexisGraphSession.BufferShape shape,
            AexisGraphSession.BufferShape target)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.packs == 1
                && shape.dims == 3
                && shape.w == target.w
                && shape.h == target.c
                && shape.d == 1
                && shape.c == 1
                && AexisGraphSession.MatchesPack4TextureStorage(tensor, shape);
        }

        internal static bool TryResolveScalarSingleBroadcastShapes(
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out AexisGraphSession.BufferShape storageShape)
        {
            broadcastMode = 0;
            outShape = default;
            storageShape = default;

            static int GetRows(AexisGraphSession.BufferShape shape) => shape.dims == 1 ? 1 : shape.h;
            static int GetCols(AexisGraphSession.BufferShape shape) => shape.w;
            static int ElementCount(AexisGraphSession.BufferShape shape) => Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            static bool IsScalarSingle(AexisGraphSession.BufferShape shape) => shape.dims >= 1 && shape.dims <= 2 && ElementCount(shape) == 1;
            static bool CanUseScalarSingleOutput(AexisGraphSession.BufferShape shape) => shape.dims == 1 || shape.dims == 2;
            static AexisGraphSession.BufferShape ScalarSingleStorage(AexisGraphSession.BufferShape shape)
            {
                var width = Mathf.Max(1, shape.w);
                var height = shape.dims == 1 ? 1 : Mathf.Max(1, shape.h);
                return new AexisGraphSession.BufferShape(3, width, height, 1, 1);
            }
            static bool IsRowVector(AexisGraphSession.BufferShape shape) => shape.dims == 2 && shape.h == 1 && shape.w > 0;
            static bool IsColumnVector(AexisGraphSession.BufferShape shape) => shape.dims == 2 && shape.w == 1 && shape.h > 0;

            var aRows = GetRows(aShape);
            var aCols = GetCols(aShape);
            var bRows = GetRows(bShape);
            var bCols = GetCols(bShape);

            // Scalar-with-scalar (element count one on both sides): the XPOS
            // relative-position metadata chain computes mel_len - k before the
            // value is broadcast into a shape vector.  Mode 0 samples both
            // operands at the single texel, so the result stays scalar.
            if (IsScalarSingle(aShape) && IsScalarSingle(bShape))
            {
                broadcastMode = 0;
                outShape = aShape;
                storageShape = ScalarSingleStorage(aShape);
                return true;
            }

            if (IsScalarSingle(aShape) && !IsScalarSingle(bShape) && CanUseScalarSingleOutput(bShape))
            {
                broadcastMode = 5;
                outShape = bShape;
                storageShape = ScalarSingleStorage(bShape);
                return true;
            }

            if (IsScalarSingle(bShape) && !IsScalarSingle(aShape) && CanUseScalarSingleOutput(aShape))
            {
                broadcastMode = 6;
                outShape = aShape;
                storageShape = ScalarSingleStorage(aShape);
                return true;
            }

            if (aShape.dims == 2 && bShape.dims == 1 && aRows == bCols && aCols > 1)
            {
                broadcastMode = 8;
                outShape = aShape;
                storageShape = new AexisGraphSession.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 1 && bShape.dims == 2 && aCols == bRows && bCols > 1)
            {
                broadcastMode = 7;
                outShape = bShape;
                storageShape = new AexisGraphSession.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 2 && bShape.dims == 1 && aCols == bCols && aRows > 1)
            {
                broadcastMode = 3;
                outShape = aShape;
                storageShape = new AexisGraphSession.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 1 && bShape.dims == 2 && aCols == bCols && bRows > 1)
            {
                broadcastMode = 1;
                outShape = bShape;
                storageShape = new AexisGraphSession.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 2 && IsRowVector(bShape) && aCols == bCols && aRows > 1)
            {
                broadcastMode = 3;
                outShape = aShape;
                storageShape = new AexisGraphSession.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (IsRowVector(aShape) && bShape.dims == 2 && aCols == bCols && bRows > 1)
            {
                broadcastMode = 1;
                outShape = bShape;
                storageShape = new AexisGraphSession.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            if (aShape.dims == 2 && IsColumnVector(bShape) && aRows == bRows && aCols > 1)
            {
                broadcastMode = 4;
                outShape = aShape;
                storageShape = new AexisGraphSession.BufferShape(3, aShape.w, aShape.h, 1, 1);
                return true;
            }

            if (IsColumnVector(aShape) && bShape.dims == 2 && aRows == bRows && bCols > 1)
            {
                broadcastMode = 2;
                outShape = bShape;
                storageShape = new AexisGraphSession.BufferShape(3, bShape.w, bShape.h, 1, 1);
                return true;
            }

            return false;
        }

        private static bool TryGetSingleElementBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out ComputeBuffer scalar)
        {
            scalar = null;
            if (string.IsNullOrEmpty(name) || bufferBlobs == null)
                return false;
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null || buffer.count < 1)
                return false;

            if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
            {
                if (view.elementCount != 1)
                    return false;
            }
            else if (buffer.count != 1)
            {
                return false;
            }

            scalar = buffer;
            return true;
        }

        private static bool TryGetChannelVectorBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisTensorBuffer vector)
        {
            vector = null;
            if (string.IsNullOrEmpty(name) || bufferBlobs == null || bufferViews == null)
                return false;
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null)
                return false;
            if (!bufferViews.TryGetValue(name, out var view) || view == null || view.buffer == null)
                return false;

            var isVector =
                view.dims == 1
                || (view.dims == 2 && (view.w == 1 || view.h == 1))
                || (view.dims == 3 && view.w == 1 && view.h == 1)
                || (view.dims == 4 && view.w == 1 && view.h == 1 && view.d == 1);
            if (!isVector)
                return false;
            if (view.elementCount <= 1)
                return false;

            vector = view;
            return true;
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var withScalar = layer.GetInt(1, 0);

            if (withScalar != 0)
            {
                return !owner.ShouldForceCurrentLayerBufferPath()
                    && AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var scalarTex, out var scalarTexShape)
                    && CanUseScalarLikeTexturePath(scalarTex, scalarTexShape);
            }

            var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                ? AexisGraphSession.CodeFormerSftAddLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
            var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                ? AexisGraphSession.CodeFormerSftMulLayers.Contains(layer.name)
                : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);

            AexisGraphSession.TensorRef aTex = null;
            AexisGraphSession.TensorRef bTex = null;
            AexisGraphSession.BufferShape aTexShape = default;
            AexisGraphSession.BufferShape bTexShape = default;
            var canUseTextureBinary = false;

            if (TryResolveExactPack4LinearBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (TryResolvePack4LinearMixedBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (TryResolveLinearMatScalarArrayBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            canUseTextureBinary = !owner.ForceBufferBinaryOpAll
                && owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out aTex, out aTexShape)
                && owner.TryGetPack4Texture(layer.bottomNames[1], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out bTex, out bTexShape);

            if (canUseTextureBinary && AexisGraphSession.CanUseExactPack4BinaryPath(aTex, aTexShape, bTex, bTexShape))
                return true;

            if (TryResolveExactScalar2DTextureBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (TryResolveScalarSingleBroadcastTextureBinaryPath(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    out _,
                     out _,
                     out _,
                     out _,
                     out _,
                     out _,
                     out _))
            {
                return true;
            }

            if (TryResolvePack4TextureScalarTexture(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4TextureChannelVectorBroadcast(
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                     context.textureShapes,
                     out _,
                     out _,
                     out _,
                     out _,
                     out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4ChannelVectorBroadcast(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            if (!owner.ForceBufferBinaryOpAll
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4BufferScalar(
                    owner,
                    layer.bottomNames[0],
                    layer.bottomNames[1],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _,
                    out _,
                    out _))
            {
                return true;
            }

            return canUseTextureBinary
                && !isTargetSftAddLayer
                && !isTargetSftMulLayer
                && TryResolvePack4SpatialBroadcast(aTex, aTexShape, bTex, bTexShape, out _, out _, out _, out _, out _);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var opType = layer.GetInt(0, 0);
                                                var withScalar = layer.GetInt(1, 0);
                                                var scalarB = layer.GetFloat(2, 0f);
                                                var isTargetSftAddLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftAddLayer)
                                                    ? AexisGraphSession.CodeFormerSftAddLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftAddLayer, StringComparison.Ordinal);
                                                var isTargetSftMulLayer = string.IsNullOrEmpty(owner.CodeFormerTargetSftMulLayer)
                                                    ? AexisGraphSession.CodeFormerSftMulLayers.Contains(layer.name)
                                                    : string.Equals(layer.name, owner.CodeFormerTargetSftMulLayer, StringComparison.Ordinal);
                                                var isCodeFormerSftMul = opType == 2 && isTargetSftMulLayer;
                                                var scalarTextureFormat = AexisGraphSession.ResolveTensorTextureFormat(2);
                                                var a = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var aShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                // [TTS-DIAG] tail binary-op 输入 nameID 日志（verbose 开关）
                                                if (AexisGraphSession.s_ttsVerbose)
                                                {
                                                    var bs = new System.Text.StringBuilder();
                                                    bs.Append($"[TTS-DIAG] BINOP {layer.name} op={opType} withScalar={withScalar} a0={layer.bottomNames[0]}({a.texture.nameID},w={a.width})");
                                                    for (int bi = 1; bi < layer.bottomNames.Length; bi++)
                                                    { try { var bb = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[bi]); bs.Append($" b{bi}={layer.bottomNames[bi]}({bb.texture.nameID})"); } catch { bs.Append($" b{bi}={layer.bottomNames[bi]}(MISSING)"); } }
                                                    UnityEngine.Debug.Log(bs.ToString());
                                                    // [TTS-DIAG] cmd.CopyTexture 到静态 staging（Add_10/Div_4 专用）
                                                    if (layer.name.Contains("/dec/Add_10") || layer.name.Contains("/dec/Div_4") || layer.name.Contains("/dec/Add_9"))
                                                    {
                                                        try
                                                        {
                                                            if (s_tailBinRt == null || s_tailBinRt.width != a.width)
                                                            {
                                                                if (s_tailBinRt != null) { s_tailBinRt.Release(); UnityEngine.Object.DestroyImmediate(s_tailBinRt); }
                                                                var dd = new RenderTextureDescriptor(a.width, a.height, a.texture.format, 0)
                                                                { dimension = TextureDimension.Tex2DArray, volumeDepth = Mathf.Max(1, a.texture.depth), enableRandomWrite = true, msaaSamples = 1 };
                                                                s_tailBinRt = new RenderTexture(dd) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
                                                                s_tailBinRt.Create();
                                                            }
                                                            var __srcTex = a.texture.directTexture;
                                                            // [TTS-FIX §nullsource] 临时 RT 无 directTexture（仅 nameID）→ 延迟执行时源为 null。
                                                            // 改用 nameID（RenderTargetIdentifier），与派发时点快照同语义，RT 存活时正常解析。
                                                            if (__srcTex != null)
                                                            {
                                                                for (int sl = 0; sl < Mathf.Min(a.texture.depth, s_tailBinRt.volumeDepth); sl++)
                                                                    cmd.CopyTexture(__srcTex, sl, 0, s_tailBinRt, sl, 0);
                                                            }
                                                            else
                                                            {
                                                                for (int sl = 0; sl < Mathf.Min(a.texture.depth, s_tailBinRt.volumeDepth); sl++)
                                                                    cmd.CopyTexture(a.texture.nameID, sl, 0, s_tailBinRt, sl, 0);
                                                            }
                                                            s_tailBinTag = layer.name;
                                                            // [TTS-DIAG] 同时快照 b 输入（Add_9/Add_10 的第二路）
                                                            try {
                                                                if (layer.bottomNames.Length > 1) {
                                                                    var bb = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                                    var bbTex = bb.texture;
                                                                    // [TTS-FIX §mismatch] b 可能为 1x1 标量常量（Div_4 mode=9 广播），
                                                                    // staging 须按 b 实际尺寸创建，否则 CopyTexture(1x1→16384x1) 报 mismatching data size。
                                                                    if (s_tailBinRtB == null
                                                                        || s_tailBinRtB.width != bbTex.width
                                                                        || s_tailBinRtB.height != bbTex.height
                                                                        || s_tailBinRtB.volumeDepth != Mathf.Max(1, bbTex.depth)) {
                                                                        if (s_tailBinRtB != null) { s_tailBinRtB.Release(); UnityEngine.Object.DestroyImmediate(s_tailBinRtB); }
                                                                        var ddb = new RenderTextureDescriptor(bbTex.width, bbTex.height, bbTex.format, 0)
                                                                        { dimension = TextureDimension.Tex2DArray, volumeDepth = Mathf.Max(1, bbTex.depth), enableRandomWrite = true, msaaSamples = 1 };
                                                                        s_tailBinRtB = new RenderTexture(ddb) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
                                                                        s_tailBinRtB.Create();
                                                                    }
                                                                    var __srcTexB = bb.texture.directTexture;
                                                                    // [TTS-FIX §nullsource] 同 tailBin：directTexture 为 null 时改走 nameID。
                                                                    if (__srcTexB != null)
                                                                    {
                                                                        for (int sl = 0; sl < Mathf.Min(bb.texture.depth, s_tailBinRtB.volumeDepth); sl++)
                                                                            cmd.CopyTexture(__srcTexB, sl, 0, s_tailBinRtB, sl, 0);
                                                                    }
                                                                    else
                                                                    {
                                                                        for (int sl = 0; sl < Mathf.Min(bb.texture.depth, s_tailBinRtB.volumeDepth); sl++)
                                                                            cmd.CopyTexture(bb.texture.nameID, sl, 0, s_tailBinRtB, sl, 0);
                                                                    }
                                                                }
                                                            } catch (System.Exception __bex) { UnityEngine.Debug.Log($"[TTS-DIAG] BINOP stage B fail {layer.name}: {__bex.Message}"); }
                                                            if (AexisGraphSession.s_ttsVerbose) UnityEngine.Debug.Log($"[TTS-DIAG] BINOP stage COPY {layer.name} tag={s_tailBinTag}");
                                                        }
                                                        catch (System.Exception __rx) { UnityEngine.Debug.Log($"[TTS-DIAG] BINOP stage fail {layer.name}: {__rx.Message}"); }
                                                    }
                                                }
                                                if (withScalar != 0)
                                                {
                                                    if (!CanUseScalarLikeTexturePath(a, aShape))
                                                        break;
                                                    var aStorageShape = AexisGraphSession.GetCmdStorageShape(a, aShape);
                                                    if (AexisGraphSession.IsPack4LinearMatTexture(a, aShape))
                                                    {
                                                        var scalarPack4Out = owner.RentTempArray(cmd, aStorageShape.w, aStorageShape.h, 1, a.texture.format);
                                                        owner.Ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, 1, opType, scalarPack4Out);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(scalarPack4Out, aShape, aStorageShape, owned: true);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                        owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                        continue;
                                                    }

                                                    if (AexisGraphSession.IsStrictLinearMatTexture(a))
                                                    {
                                                        var outLinear = owner.RentTempMat(cmd, aStorageShape.w, aStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                        owner.Ops.BinaryOpScalarLinearMat(cmd, a.texture, scalarB, opType, outLinear);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outLinear, aShape, aStorageShape, owned: true);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                        owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                        continue;
                                                    }
                                                    ComputeTexture aScalarMaterialized = null;
                                                    var aScalarInput = MaterializeScalarLikeTexture(owner, cmd, a, aShape, scalarTextureFormat, ref aScalarMaterialized);
                                                    var scalarDepth = ResolveTexturePhysicalDepth(aScalarInput, a.packs);
                                                    var outArr = owner.RentTempArray(cmd, aScalarInput.width, aScalarInput.height, scalarDepth, AexisGraphSession.ResolveTensorTextureFormat(aShape.dims));
                                                    owner.Ops.BinaryOpScalarPack4(cmd, aScalarInput, scalarB, scalarDepth, opType, outArr);
                                                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, aShape, aStorageShape, owned: true);
                                                    if (shapes != null)
                                                        shapes[layer.topNames[0]] = aShape;
                                                    if (aScalarMaterialized != null)
                                                        owner.ReturnTempArray(cmd, aScalarMaterialized);
                                                }
                                                else
                                                {
                                                    var b = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[1]);
                                                    var bShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
                                                    if (AexisGraphSession.IsPack4LinearMatTexture(a, aShape)
                                                        && AexisGraphSession.IsPack4LinearMatTexture(b, bShape)
                                                        && aShape.w == bShape.w
                                                        && aShape.h == bShape.h
                                                        && aShape.d == bShape.d
                                                        && aShape.c == bShape.c
                                                        && AexisGraphSession.BufferShapeEquals(AexisGraphSession.GetCmdStorageShape(a, aShape), AexisGraphSession.GetCmdStorageShape(b, bShape)))
                                                    {
                                                        var storageShape = AexisGraphSession.GetCmdStorageShape(a, aShape);
                                                        var outArr = owner.RentTempArray(cmd, storageShape.w, storageShape.h, 1, a.texture.format);
                                                        owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, 1, opType, outArr);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, aShape, storageShape, owned: true);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else if (TryResolveLinearMatScalarArrayBinaryPath(
                                                        blobs,
                                                        shapes,
                                                        layer.bottomNames[0],
                                                        layer.bottomNames[1],
                                                        out var scalarArrayLinear,
                                                        out var scalarArray,
                                                        out var scalarArrayShape,
                                                        out var scalarArrayStorageShape,
                                                        out var linearIsA))
                                                    {
                                                        var outLinear = owner.RentTempMat(
                                                            cmd,
                                                            scalarArrayStorageShape.w,
                                                            scalarArrayStorageShape.h,
                                                            AexisGraphSession.ResolveLinearMatTextureFormat());
                                                        owner.Ops.BinaryOpLinearMatScalarArray(
                                                            cmd,
                                                            scalarArrayLinear.texture,
                                                            scalarArray.texture,
                                                            linearIsA,
                                                            opType,
                                                            outLinear);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                                                            outLinear,
                                                            scalarArrayShape,
                                                            scalarArrayStorageShape,
                                                            owned: true,
                                                            blobName: layer.topNames[0]);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = scalarArrayShape;
                                                    }
                                                    else if (TryResolvePack4LinearMixedBinaryPath(
                                                        blobs,
                                                        shapes,
                                                        layer.bottomNames[0],
                                                        layer.bottomNames[1],
                                                        out var mixedPack4Linear,
                                                        out var mixedLinear,
                                                        out var mixedOutShape,
                                                        out var mixedStorageShape,
                                                        out var mixedPack4IsA))
                                                    {
                                                        var outArr = owner.RentTempArray(
                                                            cmd,
                                                            mixedStorageShape.w,
                                                            mixedStorageShape.h,
                                                            ResolveTexturePhysicalDepth(mixedPack4Linear.texture, mixedPack4Linear.packs),
                                                            mixedPack4Linear.texture.format);
                                                        if (mixedLinear.texture.dimension == TextureDimension.Tex2DArray)
                                                            owner.Ops.BinaryOpPack4LinearMixedArray(cmd, mixedPack4Linear.texture, mixedLinear.texture, mixedPack4IsA, opType, outArr);
                                                        else
                                                            owner.Ops.BinaryOpPack4LinearMixed(cmd, mixedPack4Linear.texture, mixedLinear.texture, mixedPack4IsA, opType, outArr);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, mixedOutShape, mixedStorageShape, owned: true);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = mixedOutShape;
                                                    }
                                                    else if (owner.IsVocoderTail8192TilingLayer(layer.name)
                                                        && aShape.w > 8192 && aShape.w % 8192 == 0
                                                        && aShape.h == 1 && a.width == 8192
                                                        && a.height == aShape.w / 8192
                                                        && ((aShape.dims == bShape.dims && aShape.w == bShape.w
                                                                && aShape.h == bShape.h && aShape.c == bShape.c
                                                                && a.width == b.width && a.height == b.height)
                                                            || (bShape.w == 1 && bShape.h == 1 && bShape.d == 1
                                                                && bShape.c == 1 && CanUseScalarLikeTexturePath(b, bShape))))
                                                    {
                                                        var outArr = owner.RentTempArray(cmd, a.width, a.height, a.packs,
                                                            owner.ResolveActivationTextureFormat(aShape.dims));
                                                        if (bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && bShape.c == 1)
                                                        {
                                                            if (b.texture.dimension == TextureDimension.Tex2D)
                                                                owner.Ops.BinaryOpPack4TexScalar(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                            else
                                                                owner.Ops.BinaryOpPack4Broadcast(cmd, a.texture, b.texture, a.packs, opType, 9, outArr);
                                                        }
                                                        else
                                                            owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                        blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, aShape,
                                                            AexisGraphSession.GetCmdStorageShape(a, aShape), owned: true);
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else if (CanUseExactCmdBinaryPath(a, aShape, b, bShape))
                                                    {
                                                        var outDepth = aShape.dims == 4 ? Mathf.Max(1, aShape.d) * a.packs : a.packs;
                                                        var outArr = owner.RentTempArray(
                                                            cmd,
                                                            a.width,
                                                            a.height,
                                                            outDepth,
                                                            owner.ResolveActivationTextureFormat(aShape.dims));
                                                        owner.Ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                        blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = a.width,
                                                            height = a.height,
                                                            packs = a.packs,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = aShape,
                                                            hasStorageShape = true,
                                                            storageShape = aShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else if (CanUseScalar2DTexturePath(a, aShape)
                                                        && CanUseScalar2DTexturePath(b, bShape)
                                                        && aShape.w == bShape.w
                                                        && aShape.h == bShape.h
                                                        && aShape.d == bShape.d
                                                        && aShape.c == bShape.c
                                                        && a.width == b.width
                                                        && a.height == b.height
                                                        && a.packs == b.packs)
                                                    {
                                                        ComputeTexture aScalarMaterialized = null;
                                                        ComputeTexture bScalarMaterialized = null;
                                                        ComputeTexture scaledB = null;
                                                        ComputeTexture finalOut = null;
                                                        try
                                                        {
                                                            if (CanUseExactLinearMatBinaryPath(a, aShape, b, bShape, out var scalar2DStorageShape))
                                                            {
                                                                var rhsLinear = b.texture;
                                                                if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                                {
                                                                    scaledB = owner.RentTempMat(cmd, scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                                    owner.Ops.BinaryOpScalarLinearMat(cmd, rhsLinear, owner.CodeFormerSftAddScale, 2, scaledB);
                                                                    rhsLinear = scaledB;
                                                                }

                                                                finalOut = owner.RentTempMat(cmd, scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                                if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                                {
                                                                    owner.Ops.BinaryOpScalarLinearMat(cmd, a.texture, 0f, 0, finalOut);
                                                                }
                                                                else
                                                                {
                                                                    owner.Ops.BinaryOpLinearMat(cmd, a.texture, rhsLinear, opType, finalOut);
                                                                    if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                    {
                                                                        var scaledOut = owner.RentTempMat(cmd, scalar2DStorageShape.w, scalar2DStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                                        owner.Ops.BinaryOpScalarLinearMat(cmd, finalOut, owner.CodeFormerSftMulScale, 2, scaledOut);
                                                                        owner.ReturnTempArray(cmd, finalOut);
                                                                        finalOut = scaledOut;
                                                                    }
                                                                }

                                                                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(finalOut, aShape, scalar2DStorageShape, owned: true);
                                                                if (shapes != null)
                                                                    shapes[layer.topNames[0]] = aShape;
                                                                finalOut = null;
                                                            }
                                                            else
                                                            {
                                                            var lhs = MaterializeScalarLikeTexture(owner, cmd, a, aShape, scalarTextureFormat, ref aScalarMaterialized);
                                                            var rhs = MaterializeScalarLikeTexture(owner, cmd, b, bShape, scalarTextureFormat, ref bScalarMaterialized);
                                                            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                            {
                                                                scaledB = owner.RentTempArray(cmd, b.width, b.height, 1, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, rhs, owner.CodeFormerSftAddScale, 1, 2, scaledB);
                                                                rhs = scaledB;
                                                            }

                                                            finalOut = owner.RentTempArray(cmd, a.width, a.height, 1, scalarTextureFormat);
                                                            if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                            {
                                                                owner.Ops.CopyPack4(cmd, lhs, 0, finalOut, 0, 1);
                                                            }
                                                            else
                                                            {
                                                                owner.Ops.BinaryOpPack4(cmd, lhs, rhs, 1, opType, finalOut);
                                                                if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                {
                                                                    var scaledOut = owner.RentTempArray(cmd, a.width, a.height, 1, scalarTextureFormat);
                                                                    owner.Ops.BinaryOpScalarPack4(cmd, finalOut, owner.CodeFormerSftMulScale, 1, 2, scaledOut);
                                                                    owner.ReturnTempArray(cmd, finalOut);
                                                                    finalOut = scaledOut;
                                                                }
                                                            }

                                                            var scalarPack4StorageShape = new AexisGraphSession.BufferShape(
                                                                3,
                                                                finalOut.width,
                                                                finalOut.height,
                                                                1,
                                                                1);
                                                            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                                                                finalOut,
                                                                aShape,
                                                                scalarPack4StorageShape,
                                                                owned: true);
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = aShape;
                                                            finalOut = null;
                                                            }
                                                        }
                                                        finally
                                                        {
                                                            ReturnTempUnique(owner, cmd, ref bScalarMaterialized, aScalarMaterialized, null);
                                                            if (aScalarMaterialized != null)
                                                                owner.ReturnTempArray(cmd, aScalarMaterialized);
                                                            if (scaledB != null)
                                                                owner.ReturnTempArray(cmd, scaledB);
                                                            if (finalOut != null)
                                                                owner.ReturnTempArray(cmd, finalOut);
                                                        }
                                                    }
                                                    else if (TryResolveScalarSingleBroadcastCmdBinaryPath(
                                                        a,
                                                        aShape,
                                                        b,
                                                        bShape,
                                                        out var scalarBroadCmdA,
                                                        out var scalarBroadCmdB,
                                                        out var scalarBroadcastCmdMode,
                                                        out var scalarBroadcastCmdOutShape,
                                                        out var scalarBroadcastCmdStorageShape))
                                                    {
                                                        ComputeTexture scalarBroadCmdAMaterialized = null;
                                                        ComputeTexture scalarBroadCmdBMaterialized = null;
                                                        ComputeTexture scaledB = null;
                                                        ComputeTexture finalOut = null;
                                                        try
                                                        {
                                                            var lhs = MaterializeScalarLikeTexture(owner, cmd, scalarBroadCmdA, aShape, scalarTextureFormat, ref scalarBroadCmdAMaterialized);
                                                            var rhs = MaterializeScalarLikeTexture(owner, cmd, scalarBroadCmdB, bShape, scalarTextureFormat, ref scalarBroadCmdBMaterialized);
                                                            if (owner.CodeFormerSftAddScale != 1f && opType == 0 && isTargetSftAddLayer)
                                                            {
                                                                scaledB = owner.RentTempArray(cmd, scalarBroadCmdB.width, scalarBroadCmdB.height, 1, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, rhs, owner.CodeFormerSftAddScale, 1, 2, scaledB);
                                                                rhs = scaledB;
                                                            }

                                                            finalOut = owner.RentTempArray(cmd, scalarBroadcastCmdStorageShape.w, scalarBroadcastCmdStorageShape.h, 1, scalarTextureFormat);
                                                            if (isCodeFormerSftMul && owner.CodeFormerBypassSftMul)
                                                            {
                                                                var zeroTexture = owner.RentTempArray(cmd, scalarBroadCmdB.width, scalarBroadCmdB.height, 1, scalarTextureFormat);
                                                                owner.Ops.BinaryOpScalarPack4(cmd, rhs, 0f, 1, 2, zeroTexture);
                                                                owner.Ops.BinaryOpScalarSingleBroadcast(
                                                                    cmd,
                                                                    lhs,
                                                                    zeroTexture,
                                                                    scalarBroadcastCmdStorageShape.w,
                                                                    scalarBroadcastCmdStorageShape.h,
                                                                    0,
                                                                    scalarBroadcastCmdMode,
                                                                    finalOut);
                                                                owner.ReturnTempArray(cmd, zeroTexture);
                                                            }
                                                            else
                                                            {
                                                                owner.Ops.BinaryOpScalarSingleBroadcast(
                                                                    cmd,
                                                                    lhs,
                                                                    rhs,
                                                                    scalarBroadcastCmdStorageShape.w,
                                                                    scalarBroadcastCmdStorageShape.h,
                                                                    opType,
                                                                    scalarBroadcastCmdMode,
                                                                    finalOut);
                                                                if (isCodeFormerSftMul && owner.CodeFormerSftMulScale != 1f)
                                                                {
                                                                    var scaledOut = owner.RentTempArray(cmd, finalOut.width, finalOut.height, 1, scalarTextureFormat);
                                                                    owner.Ops.BinaryOpScalarPack4(cmd, finalOut, owner.CodeFormerSftMulScale, 1, 2, scaledOut);
                                                                    owner.ReturnTempArray(cmd, finalOut);
                                                                    finalOut = scaledOut;
                                                                }
                                                            }

                                                            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                            {
                                                                texture = finalOut,
                                                                width = scalarBroadcastCmdStorageShape.w,
                                                                height = scalarBroadcastCmdStorageShape.h,
                                                                packs = 1,
                                                                refs = 1,
                                                                owned = true,
                                                                hasLogicalShape = true,
                                                                logicalShape = scalarBroadcastCmdOutShape,
                                                                hasStorageShape = true,
                                                                storageShape = scalarBroadcastCmdStorageShape
                                                            };
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = scalarBroadcastCmdOutShape;
                                                            finalOut = null;
                                                        }
                                                        finally
                                                        {
                                                            ReturnTempUnique(owner, cmd, ref scalarBroadCmdBMaterialized, scalarBroadCmdAMaterialized, null);
                                                            if (scalarBroadCmdAMaterialized != null)
                                                                owner.ReturnTempArray(cmd, scalarBroadCmdAMaterialized);
                                                            if (scaledB != null)
                                                                owner.ReturnTempArray(cmd, scaledB);
                                                            if (finalOut != null)
                                                                owner.ReturnTempArray(cmd, finalOut);
                                                        }
                                                    }
                                                    else if (!owner.ForceBufferBinaryOpAll
                                                        && !isTargetSftAddLayer
                                                        && !isTargetSftMulLayer
                                                        && TryResolveCmdPack4TextureChannelVectorBroadcast(
                                                            a,
                                                            aShape,
                                                            b,
                                                            bShape,
                                                            out var vectorCmdTexture,
                                                            out var vectorCmdTextureShape,
                                                            out var channelVectorCmdTexture,
                                                            out var channelVectorCmdTextureShape,
                                                            out var channelVectorCmdIsA))
                                                    {
                                                        ComputeTexture channelVectorCmdMaterialized = null;
                                                        try
                                                        {
                                                            var channelVectorCmdInput = MaterializeScalarLikeTexture(
                                                                owner,
                                                                cmd,
                                                                channelVectorCmdTexture,
                                                                channelVectorCmdTextureShape,
                                                                scalarTextureFormat,
                                                                ref channelVectorCmdMaterialized,
                                                                AexisGraphSession.GetStrictPack4ScratchIdentity(
                                                                    layer,
                                                                    "binary-channel-vector-materialized"));
                                                            var outDepth = vectorCmdTextureShape.dims == 4
                                                                ? Mathf.Max(1, vectorCmdTextureShape.d) * vectorCmdTexture.packs
                                                                : vectorCmdTexture.packs;
                                                            var outArr = owner.RentTempArray(cmd, vectorCmdTexture.width, vectorCmdTexture.height, outDepth, vectorCmdTexture.texture.format);
                                                            owner.Ops.BinaryOpPack4ChannelVectorTex(
                                                                cmd,
                                                                vectorCmdTexture.texture,
                                                                channelVectorCmdInput,
                                                                vectorCmdTexture.packs,
                                                                opType,
                                                                channelVectorCmdIsA,
                                                                outArr);
                                                            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                            {
                                                                texture = outArr,
                                                                width = vectorCmdTexture.width,
                                                                height = vectorCmdTexture.height,
                                                                packs = vectorCmdTexture.packs,
                                                                refs = 1,
                                                                owned = true,
                                                                hasLogicalShape = true,
                                                                logicalShape = vectorCmdTextureShape,
                                                                hasStorageShape = true,
                                                                storageShape = vectorCmdTextureShape
                                                            };
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = vectorCmdTextureShape;
                                                        }
                                                        finally
                                                        {
                                                            if (channelVectorCmdMaterialized != null)
                                                                owner.ReturnTempArray(cmd, channelVectorCmdMaterialized);
                                                        }
                                                    }
                                                    else if (!owner.ForceBufferBinaryOpAll
                                                        && !isTargetSftAddLayer
                                                        && !isTargetSftMulLayer
                                                        && TryResolveCmdPack4ChannelLayoutReshape(
                                                        a,
                                                        aShape,
                                                        b,
                                                        bShape,
                                                        out var channelLayoutTarget,
                                                        out var channelLayoutTargetShape,
                                                        out var channelLayoutSource,
                                                        out var channelLayoutSourceShape,
                                                        out var channelLayoutSourceIsA))
                                                    {
                                                        ComputeTexture reshapedSource = null;
                                                        ComputeTexture finalOut = null;
                                                        try
                                                        {
                                                            reshapedSource = owner.RentTempArray(
                                                                cmd,
                                                                channelLayoutTarget.width,
                                                                channelLayoutTarget.height,
                                                                channelLayoutTarget.packs,
                                                                channelLayoutTarget.texture.format,
                                                                AexisGraphSession.GetStrictPack4ScratchIdentity(
                                                                    layer,
                                                                    "binary-channel-layout-reshape"));
                                                            owner.Ops.ReshapePack4ToPack4(
                                                                cmd,
                                                                channelLayoutSource.texture,
                                                                channelLayoutSourceShape.w,
                                                                channelLayoutSourceShape.h,
                                                                channelLayoutSourceShape.d,
                                                                channelLayoutSourceShape.c,
                                                                channelLayoutSourceShape.dims,
                                                                channelLayoutTargetShape.w,
                                                                channelLayoutTargetShape.h,
                                                                channelLayoutTargetShape.d,
                                                                channelLayoutTargetShape.c,
                                                                channelLayoutTargetShape.dims,
                                                                reshapedSource);

                                                            finalOut = owner.RentTempArray(
                                                                cmd,
                                                                channelLayoutTarget.width,
                                                                channelLayoutTarget.height,
                                                                channelLayoutTarget.packs,
                                                                channelLayoutTarget.texture.format);
                                                            var lhs = channelLayoutSourceIsA ? reshapedSource : channelLayoutTarget.texture;
                                                            var rhs = channelLayoutSourceIsA ? channelLayoutTarget.texture : reshapedSource;
                                                            owner.Ops.BinaryOpPack4(cmd, lhs, rhs, channelLayoutTarget.packs, opType, finalOut);
                                                            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                            {
                                                                texture = finalOut,
                                                                width = channelLayoutTarget.width,
                                                                height = channelLayoutTarget.height,
                                                                packs = channelLayoutTarget.packs,
                                                                refs = 1,
                                                                owned = true,
                                                                hasLogicalShape = true,
                                                                logicalShape = channelLayoutTargetShape,
                                                                hasStorageShape = true,
                                                                storageShape = channelLayoutTargetShape
                                                            };
                                                            if (shapes != null)
                                                                shapes[layer.topNames[0]] = channelLayoutTargetShape;
                                                            finalOut = null;
                                                        }
                                                        finally
                                                        {
                                                            if (reshapedSource != null)
                                                                owner.ReturnTempArray(cmd, reshapedSource);
                                                            if (finalOut != null)
                                                                owner.ReturnTempArray(cmd, finalOut);
                                                        }
                                                    }
                                                    else if (TryResolveCmdSpatialBroadcast(a, aShape, b, bShape, out var broadcastMode, out var outShape, out var outWidth, out var outHeight, out var outPacks))
                                                    {
                                                        var outDepth = outShape.dims == 4 ? Mathf.Max(1, outShape.d) * outPacks : outPacks;
                                                        var outArr = owner.RentTempArray(cmd, outWidth, outHeight, outDepth, RenderTextureFormat.ARGBHalf);
                                                        owner.Ops.BinaryOpPack4Broadcast(cmd, a.texture, b.texture, outPacks, opType, broadcastMode, outArr);
                                                        blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = outWidth,
                                                            height = outHeight,
                                                            packs = outPacks,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = outShape,
                                                            hasStorageShape = true,
                                                            storageShape = outShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = outShape;
                                                    }
                                                    else if (aShape.dims == 3
                                                        && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                                                        && bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && bShape.c == 1
                                                        && CanUseScalarLikeTexturePath(b, bShape))
                                                    {
                                                        // d3 Pack4 激活 / d1 单元素标量纹理：标量恒采 b[0,0,0].x 全广播。
                                                        // [FIX] MemoryData 单元素常量（Constant_4 等）以 Tex2D（维度2, RFloat）发布，
                                                        //       直接绑 _TexIn1Arr（Texture2DArray）触发 "expected 5, got 2" 维度不匹配
                                                        //       → 内核执行未定义 → Div_4 输出零 → 下游全零（静音 WAV）。
                                                        //       方案：b 为 Tex2D 时改用专用内核 AexisBinaryOpPack4TexScalar，
                                                        //       直接从 _LinearIn1[0,0] 读标量（不租新 RT、无计划器冲突、零回读）。
                                                        var outDepth = a.packs;
                                                        var outArr = owner.RentTempArray(cmd, a.width, a.height, outDepth, RenderTextureFormat.ARGBHalf);
                                                        if (b.texture != null && b.texture.dimension == TextureDimension.Tex2D)
                                                            owner.Ops.BinaryOpPack4TexScalar(cmd, a.texture, b.texture, a.packs, opType, outArr);
                                                        else
                                                            owner.Ops.BinaryOpPack4Broadcast(cmd, a.texture, b.texture, a.packs, opType, 9, outArr);
                                                        blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                                                        {
                                                            texture = outArr,
                                                            width = a.width,
                                                            height = a.height,
                                                            packs = a.packs,
                                                            refs = 1,
                                                            owned = true,
                                                            hasLogicalShape = true,
                                                            logicalShape = aShape,
                                                            hasStorageShape = true,
                                                            storageShape = aShape
                                                        };
                                                        if (shapes != null)
                                                            shapes[layer.topNames[0]] = aShape;
                                                    }
                                                    else
                                                    {
                                                        var fallbackShape = ResolveCmdOutputShape(aShape, bShape);
                                                        owner.DebugLog?.Invoke(
                                                            "[CmdPlaceholder][BinaryOp]"
                                                            + " | layer=" + layer.name
                                                            + " | opType=" + opType.ToString(CultureInfo.InvariantCulture)
                                                            + " | resizeGuard=" + owner.DisallowBufferAccess.ToString()
                                                            + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                                                            + " | aTex=" + a.width + "x" + a.height + "x" + a.packs
                                                            + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                                                            + " | bTex=" + b.width + "x" + b.height + "x" + b.packs
                                                            + " | out=d" + fallbackShape.dims + ":" + fallbackShape.w + "x" + fallbackShape.h + "x" + fallbackShape.d + "x" + fallbackShape.c);
                                                        owner.DebugLog?.Invoke(
                                                            "[CmdPlaceholder][BinaryOpShape]"
                                                            + " | layer=" + layer.name
                                                            + " | aStorage=" + AexisGraphSession.GetCmdStorageShape(a, aShape)
                                                            + " | bStorage=" + AexisGraphSession.GetCmdStorageShape(b, bShape)
                                                            + " | aLogical=" + aShape
                                                            + " | bLogical=" + bShape
                                                            + " | outLogical=" + fallbackShape);
                                                        throw new InvalidOperationException(
                                                            "BinaryOp command-buffer Pack4 broadcast profile is unsupported"
                                                            + " | layer=" + layer.name
                                                            + " | opType=" + opType.ToString(CultureInfo.InvariantCulture)
                                                            + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                                                            + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                                                            + " | out=d" + fallbackShape.dims + ":" + fallbackShape.w + "x" + fallbackShape.h + "x" + fallbackShape.d + "x" + fallbackShape.c
                                                            + " | rejectedFallback=placeholder");
                                                    }
                                                }
                                                // [TTS-DIAG] 快照 BinaryOp 输出（FP16 断点定位，verbose 开关）
                                                if (AexisGraphSession.s_ttsVerbose && (layer.name.Contains("/dec/Add_") || layer.name.Contains("/dec/Div_")))
                                                {
                                                    try
                                                    {
                                                        var oT = AexisGraphSession.GetCmdTensor(blobs, layer.topNames[0]);
                                                        var oTex = oT.texture;
                                                        if (oTex == null) { UnityEngine.Debug.Log($"[TTS-DIAG] BINOP-OUT {layer.name} OUT NULL"); }
                                                        else
                                                        {
                                                            if (s_tailOutRt == null || s_tailOutRt.width != oTex.width || s_tailOutRt.height != oTex.height || s_tailOutRt.volumeDepth != Mathf.Max(1, oTex.depth) || s_tailOutRt.format != oTex.format)
                                                            {
                                                                if (s_tailOutRt != null) { s_tailOutRt.Release(); UnityEngine.Object.DestroyImmediate(s_tailOutRt); }
                                                                var ddO = new RenderTextureDescriptor(oTex.width, oTex.height, oTex.format, 0)
                                                                { dimension = TextureDimension.Tex2DArray, volumeDepth = Mathf.Max(1, oTex.depth), enableRandomWrite = true, msaaSamples = 1 };
                                                                s_tailOutRt = new RenderTexture(ddO) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
                                                                s_tailOutRt.Create();
                                                            }
                                                            var __srcO = oTex.directTexture;
                                                            for (int sl = 0; sl < Mathf.Min(oTex.depth, s_tailOutRt.volumeDepth); sl++)
                                                            {
                                                                if (__srcO != null) cmd.CopyTexture(__srcO, sl, 0, s_tailOutRt, sl, 0);
                                                                else cmd.CopyTexture(oTex.nameID, sl, 0, s_tailOutRt, sl, 0);
                                                            }
                                                            s_tailOutTag = layer.name;
                                                        }
                                                    }
                                                    catch (Exception ex) { UnityEngine.Debug.Log($"[TTS-DIAG] BINOP-OUT {layer.name} EX={ex.Message}"); }
                                                }
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanUseExactCmdBinaryPath(AexisGraphSession.CmdTensorRef a, AexisGraphSession.BufferShape aShape, AexisGraphSession.CmdTensorRef b, AexisGraphSession.BufferShape bShape)
        {
            return a != null
                && b != null
                && a.texture != null
                && b.texture != null
                && (aShape.dims == 3 || aShape.dims == 4)
                && aShape.dims == bShape.dims
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape)
                && a.width == b.width
                && a.height == b.height
                && a.packs == b.packs;
        }

        private static bool TryResolveCmdSpatialBroadcast(
            AexisGraphSession.CmdTensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef b,
            AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (a == null || b == null || a.texture == null || b.texture == null)
                return false;
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;

            var matchingSpatialExtent = aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d;
            if (matchingSpatialExtent
                && aShape.c == 1
                && bShape.c > 1
                && a.packs == 1
                && b.packs == Mathf.Max(1, Mathf.CeilToInt(bShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape))
            {
                broadcastMode = 3;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingSpatialExtent
                && bShape.c == 1
                && aShape.c > 1
                && b.packs == 1
                && a.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape))
            {
                broadcastMode = 4;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            if (TryResolveCmdPack4WidthVectorBroadcast(
                    a,
                    aShape,
                    b,
                    bShape,
                    out broadcastMode,
                    out outShape,
                    out outWidth,
                    out outHeight,
                    out outPacks))
            {
                return true;
            }

            var matchingRowsAndChannels = aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && a.packs == b.packs
                && a.height == b.height
                && a.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape);
            if (matchingRowsAndChannels
                && aShape.w == 1
                && bShape.w > 1
                && a.width == 1
                && b.width == bShape.w)
            {
                broadcastMode = 5;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingRowsAndChannels
                && bShape.w == 1
                && aShape.w > 1
                && b.width == 1
                && a.width == aShape.w)
            {
                broadcastMode = 6;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            // 深度轴广播：空间尺寸与 Pack4 通道打包完全一致，仅逻辑深度轴一侧为 1
            // （batch 共享的相对位置注意力项）。输出取存活一侧的深度，内核按 pack
            // 索引复用单层操作数。
            var matchingPackedSpatial = aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.c == bShape.c
                && a.packs == b.packs
                && a.packs == Mathf.Max(1, Mathf.CeilToInt(aShape.c / 4f))
                && a.width == b.width
                && a.height == b.height
                && AexisGraphSession.MatchesPack4TextureStorage(a, aShape)
                && AexisGraphSession.MatchesPack4TextureStorage(b, bShape);
            if (matchingPackedSpatial && aShape.d == 1 && bShape.d > 1)
            {
                broadcastMode = 11;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (matchingPackedSpatial && bShape.d == 1 && aShape.d > 1)
            {
                broadcastMode = 10;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            if (aShape.c != bShape.c || a.packs != b.packs)
                return false;

            var aIsScalarSpatial = aShape.w == 1 && aShape.h == 1 && aShape.d == 1 && a.width == 1 && a.height == 1;
            var bIsScalarSpatial = bShape.w == 1 && bShape.h == 1 && bShape.d == 1 && b.width == 1 && b.height == 1;
            var aIsOutputSpatial = aShape.w == a.width && aShape.h == a.height;
            var bIsOutputSpatial = bShape.w == b.width && bShape.h == b.height;

            if (aIsScalarSpatial && !bIsScalarSpatial && bIsOutputSpatial)
            {
                broadcastMode = 1;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (bIsScalarSpatial && !aIsScalarSpatial && aIsOutputSpatial)
            {
                broadcastMode = 2;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            return false;
        }

        private static bool TryResolveCmdPack4WidthVectorBroadcast(
            AexisGraphSession.CmdTensorRef a,
            AexisGraphSession.BufferShape aShape,
            AexisGraphSession.CmdTensorRef b,
            AexisGraphSession.BufferShape bShape,
            out int broadcastMode,
            out AexisGraphSession.BufferShape outShape,
            out int outWidth,
            out int outHeight,
            out int outPacks)
        {
            broadcastMode = 0;
            outShape = default;
            outWidth = 0;
            outHeight = 0;
            outPacks = 0;

            if (a == null || b == null || a.texture == null || b.texture == null)
                return false;
            if ((aShape.dims != 3 && aShape.dims != 4) || aShape.dims != bShape.dims)
                return false;

            if (IsCmdPack4WidthVector(a, aShape)
                && IsCmdPack4WidthVectorTarget(b, bShape, aShape.w))
            {
                broadcastMode = 7;
                outWidth = b.width;
                outHeight = b.height;
                outPacks = b.packs;
                outShape = new AexisGraphSession.BufferShape(bShape.dims, bShape.w, bShape.h, bShape.d, bShape.c);
                return true;
            }

            if (IsCmdPack4WidthVector(b, bShape)
                && IsCmdPack4WidthVectorTarget(a, aShape, bShape.w))
            {
                broadcastMode = 8;
                outWidth = a.width;
                outHeight = a.height;
                outPacks = a.packs;
                outShape = new AexisGraphSession.BufferShape(aShape.dims, aShape.w, aShape.h, aShape.d, aShape.c);
                return true;
            }

            return false;
        }

        private static bool IsCmdPack4WidthVector(AexisGraphSession.CmdTensorRef texture, AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h == 1
                && shape.d == 1
                && shape.c == 1
                && texture.width == shape.w
                && texture.height == 1
                && texture.packs == 1
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static bool IsCmdPack4WidthVectorTarget(AexisGraphSession.CmdTensorRef texture, AexisGraphSession.BufferShape shape, int width)
        {
            return texture != null
                && texture.texture != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w == width
                && shape.h > 0
                && shape.d > 0
                && shape.c > 0
                && texture.width == width
                && texture.packs == Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f))
                && AexisGraphSession.MatchesPack4TextureStorage(texture, shape);
        }

        private static AexisGraphSession.BufferShape ResolveCmdOutputShape(AexisGraphSession.BufferShape aShape, AexisGraphSession.BufferShape bShape)
        {
            var aCount = Mathf.Max(1, aShape.w * aShape.h * aShape.d * aShape.c);
            var bCount = Mathf.Max(1, bShape.w * bShape.h * bShape.d * bShape.c);
            if (aCount == bCount)
                return aShape;
            if (aCount < bCount && bCount % aCount == 0)
                return bShape;
            if (bCount < aCount && aCount % bCount == 0)
                return aShape;
            return aShape;
        }
    }
}
