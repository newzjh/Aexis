// Auto-generated kernel implementation group: AexisKernels.Pack4Conv.hlsl

float4 AexisApplyFusedConvPRelu(float4 value, int outputPack);
float AexisReadRawConvWeight(int index, int outputChannel);

float4 AexisReadConvWeight4(int index)
{
    if (_UseFp16ConvWeights == 0)
        return _ConvW4[index];

    uint2 packed = _ConvW4Fp16[index];
    return float4(
        f16tof32(packed.x & 0xffffu),
        f16tof32(packed.x >> 16),
        f16tof32(packed.y & 0xffffu),
        f16tof32(packed.y >> 16));
}

// StyleGAN/GFPGAN changes convolution weights per image. Keep the existing
// Pack4 index ABI, but source execution-time weights from a Texture2DArray so
// model execution never relies on a temporary ComputeBuffer.
float4 AexisReadPack4ConvWeight4(int index)
{
    if (_UseInt8ConvPack4Weights != 0)
    {
        uint packed = _ConvW4Int8Packed[index];
        uint x = packed & 0xffu;
        uint y = (packed >> 8) & 0xffu;
        uint z = (packed >> 16) & 0xffu;
        uint w = packed >> 24;
        return float4(
            x >= 128u ? (int)x - 256 : (int)x,
            y >= 128u ? (int)y - 256 : (int)y,
            z >= 128u ? (int)z - 256 : (int)z,
            w >= 128u ? (int)w - 256 : (int)w) * _Int8ConvPack4WeightScale;
    }
    if (_UseGfpganDynWeights == 0)
        return AexisReadConvWeight4(index);

    int x = index % _GfpganDynamicConvW4Width;
    int y = index / _GfpganDynamicConvW4Width;
    return _GfpganDynamicConvW4Tex[int3(x, y, 0)];
}

// The generic Pack4 kernel can vectorize regular convolutions while reading
// the canonical scalar OIHW INT8 upload. This is intentionally separate from
// the compact O4I4K SRV so 5x5 validation does not depend on a second weight
// layout. It also preserves incomplete input packs, such as SESR's RGB head.
float4 AexisReadGenericConvWeight4(
    int packedIndex,
    int outputChannel,
    int inputChannelBase,
    int kernelIndex,
    int kernelArea)
{
    if (_UseInt8ConvWeights == 0)
        return AexisReadPack4ConvWeight4(packedIndex);

    float4 value = 0.0;
    if (inputChannelBase + 0 < _InC)
        value.x = AexisReadRawConvWeight((outputChannel * _InC + inputChannelBase + 0) * kernelArea + kernelIndex, outputChannel);
    if (inputChannelBase + 1 < _InC)
        value.y = AexisReadRawConvWeight((outputChannel * _InC + inputChannelBase + 1) * kernelArea + kernelIndex, outputChannel);
    if (inputChannelBase + 2 < _InC)
        value.z = AexisReadRawConvWeight((outputChannel * _InC + inputChannelBase + 2) * kernelArea + kernelIndex, outputChannel);
    if (inputChannelBase + 3 < _InC)
        value.w = AexisReadRawConvWeight((outputChannel * _InC + inputChannelBase + 3) * kernelArea + kernelIndex, outputChannel);
    return value;
}

float4 AexisReadDepthWiseConvWeight4(int index)
{
    if (_UseFp16DepthWiseWeights == 0)
        return _DwConvW4[index];

    uint2 packed = _DwConvW4Fp16[index];
    return float4(
        f16tof32(packed.x & 0xffffu),
        f16tof32(packed.x >> 16),
        f16tof32(packed.y & 0xffffu),
        f16tof32(packed.y >> 16));
}

float4 AexisMaskConvOutputTail(float4 value, int outputPack)
{
    int firstChannel = outputPack * 4;
    if (firstChannel + 0 >= _OutC) value.x = 0.0;
    if (firstChannel + 1 >= _OutC) value.y = 0.0;
    if (firstChannel + 2 >= _OutC) value.z = 0.0;
    if (firstChannel + 3 >= _OutC) value.w = 0.0;
    return value;
}

float AexisReadPackedConvInt8(int index)
{
    uint packed = _ConvWInt8Packed[index >> 2];
    uint raw = (packed >> ((index & 3) * 8)) & 0xffu;
    int signedValue = raw >= 128u ? (int)raw - 256 : (int)raw;
    return (float)signedValue;
}

float AexisReadPackedConvInt4(int index)
{
    uint packed = _ConvWInt4Packed[index >> 3];
    uint raw = (packed >> ((index & 7) * 4)) & 0xfu;
    int signedValue = raw >= 8u ? (int)raw - 16 : (int)raw;
    return (float)signedValue;
}

// Raw OIHW is only used by the group/tail-safe CommandBuffer Conv kernels. The
// output-channel scale is applied during the texture dispatch, never by expanding
// the immutable INT8 payload to a temporary float buffer.
float AexisReadRawConvWeight(int index, int outputChannel)
{
    if (_UseInt4ConvWeights != 0)
        return AexisReadPackedConvInt4(index) * _ConvWInt4Scales[outputChannel];
    if (_UseInt8ConvWeights == 0)
        return _ConvW[index];
    float scale = _UseInt8ConvWeightUniformScale != 0
        ? _Int8ConvWeightUniformScale
        : _ConvWInt8Scales[outputChannel];
    return AexisReadPackedConvInt8(index) * scale;
}

float4 AexisReadRawConvWeight4(int outputChannelBase, int inputChannelBase, int kernelIndex)
{
    int kernelArea = 25;
    float4 result = 0.0;
    if (inputChannelBase + 0 < _InC)
    {
        result.x = AexisReadRawConvWeight(
            ((outputChannelBase * _InC + inputChannelBase + 0) * kernelArea) + kernelIndex,
            outputChannelBase);
    }
    if (inputChannelBase + 1 < _InC)
    {
        result.y = AexisReadRawConvWeight(
            ((outputChannelBase * _InC + inputChannelBase + 1) * kernelArea) + kernelIndex,
            outputChannelBase);
    }
    if (inputChannelBase + 2 < _InC)
    {
        result.z = AexisReadRawConvWeight(
            ((outputChannelBase * _InC + inputChannelBase + 2) * kernelArea) + kernelIndex,
            outputChannelBase);
    }
    if (inputChannelBase + 3 < _InC)
    {
        result.w = AexisReadRawConvWeight(
            ((outputChannelBase * _InC + inputChannelBase + 3) * kernelArea) + kernelIndex,
            outputChannelBase);
    }
    return result;
}

float AexisQuantizeActivationForInt8(float value)
{
    if (_UseInt8Activations == 0)
        return value;
    int quantized = (int)round(value / _Int8ActivationScale) + _Int8ActivationZeroPoint;
    quantized = _Int8ActivationUnsigned != 0 ? clamp(quantized, 0, 255) : clamp(quantized, -128, 127);
    return (float)(quantized - _Int8ActivationZeroPoint) * _Int8ActivationScale;
}

// Matches numpy.pad(mode="reflect") for the two-stage SESR tiling contract:
// trailing-edge padding to a whole core grid followed by the context border.
int AexisReflectIndex(int index, int extent)
{
    int n = max(1, extent);
    if (n == 1) return 0;
    int period = 2 * n - 2;
    int folded = index % period;
    if (folded < 0) folded += period;
    return folded < n ? folded : period - folded;
}

int AexisResolveTiledReflectIndex(int index, int originalExtent, int paddedExtent)
{
    int padded = AexisReflectIndex(index, max(1, paddedExtent));
    return AexisReflectIndex(padded, max(1, originalExtent));
}

void AexisPackRgbToPack4_Impl(uint3 id)
{
    uint w, h, d;
    _AexisOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    uint tw, th;
    _AexisIn.GetDimensions(tw, th);
    // Convert the unsigned dispatch coordinates before applying a negative
    // reflect offset.  Casting after the uint+int expression underflows at the
    // left/top context border on D3D11.
    int sx = (int)(((int)id.x + _OffsetX) * _ScaleX);
    int sy = (int)(((int)id.y + _OffsetY) * _ScaleY);
    sx = _InputReflectPad != 0
        ? AexisResolveTiledReflectIndex(sx, (int)tw, _InputPaddedWidth)
        : clamp(sx, 0, (int)tw - 1);
    sy = _InputReflectPad != 0
        ? AexisResolveTiledReflectIndex(sy, (int)th, _InputPaddedHeight)
        : clamp(sy, 0, (int)th - 1);
    if (_FlipY != 0)
        sy = (int)th - 1 - sy;
    float4 v = _AexisIn[int2(sx, sy)];
    v.xyz = v.xyz * _InputValueScale + _InputValueBias;
    _AexisOutArr[int3((int)id.x, (int)id.y, 0)] = float4(v.x, v.y, v.z, 0.0);
}

void AexisPackRgbToNhwcPack4_Impl(uint3 id)
{
    uint w, h, d;
    _AexisOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d) return;

    uint tw, th;
    _AexisIn.GetDimensions(tw, th);
    int rgbChannel = (int)id.x;
    // `id.y` is uint. Cast before the negative SESR context offset so the
    // NHWC reference pack and the direct NCHW pack sample identical reflected
    // pixels at the left image border.
    int sx = (int)(((int)id.y + _OffsetX) * _ScaleX);
    sx = _InputReflectPad != 0
        ? AexisResolveTiledReflectIndex(sx, (int)tw, _InputPaddedWidth)
        : clamp(sx, 0, (int)tw - 1);

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int logicalY = (int)id.z * 4 + lane;
        // `logicalY` addresses the Pack4 *destination* tensor, which can be
        // taller than the source when SESR adds reflect context.  Do not clip
        // it against `th` (the unpadded source height): that previously left
        // the lower reflect border uninitialised in whole-frame inference.
        // The destination array bounds `id.z`; any storage-only tail lanes are
        // outside the tensor shape and are never consumed by the convolution.
        int sy = (int)((logicalY + _OffsetY) * _ScaleY);
        sy = _InputReflectPad != 0
            ? AexisResolveTiledReflectIndex(sy, (int)th, _InputPaddedHeight)
            : clamp(sy, 0, (int)th - 1);
        if (_FlipY != 0)
            sy = (int)th - 1 - sy;

        float4 v = _AexisIn[int2(sx, sy)];
        float value = rgbChannel == 0 ? v.x : (rgbChannel == 1 ? v.y : v.z);
        AexisWriteLane(o, lane, value * _InputValueScale + _InputValueBias);
    }

    _AexisOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = o;
}

void AexisPackMaskToNhwcPack4_Impl(uint3 id)
{
    uint w, h, d;
    _AexisOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h || id.z >= d) return;

    uint tw, th;
    _AexisIn.GetDimensions(tw, th);
    int sx = (int)((id.y + _OffsetX) * _ScaleX);
    sx = clamp(sx, 0, (int)tw - 1);

    float4 o = 0.0;
    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int logicalY = (int)id.z * 4 + lane;
        if (logicalY >= (int)th)
            continue;

        int sy = (int)((logicalY + _OffsetY) * _ScaleY);
        sy = clamp(sy, 0, (int)th - 1);
        if (_FlipY != 0)
            sy = (int)th - 1 - sy;

        float4 v = _AexisIn[int2(sx, sy)];
        float m = max(v.x, max(v.y, v.z));
        AexisWriteLane(o, lane, m >= _MaskThreshold ? _InputValueScale + _InputValueBias : _InputValueBias);
    }

    _AexisOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = o;
}

void AexisUnpackPack4ToRgb_Impl(uint3 id)
{
    uint w, h;
    _AexisOut.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    float4 v = _AexisInArr[int3((int)id.x, (int)id.y, 0)];
    _AexisOut[int2((int)id.x, (int)id.y)] = float4(v.x, v.y, v.z, 1.0);
}

void AexisConv3x3Pack4_Impl(uint3 id, uint3 groupId, uint3 groupThreadId)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    bool active = ox < (int)ow && oy < (int)oh && op >= 0 && op < _OutPacks;
    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;

    AEXIS_CONV_VECTOR bias0 = active ? _ConvB4[op] : 0.0;
    AEXIS_CONV_VECTOR bias1 = hasOp1 ? _ConvB4[op1] : 0.0;
    AEXIS_CONV_VECTOR sum00 = bias0;
    AEXIS_CONV_VECTOR sum01 = bias0;
    AEXIS_CONV_VECTOR sum10 = bias0;
    AEXIS_CONV_VECTOR sum11 = bias0;
    AEXIS_CONV_VECTOR sum20 = bias1;
    AEXIS_CONV_VECTOR sum21 = bias1;
    AEXIS_CONV_VECTOR sum30 = bias1;
    AEXIS_CONV_VECTOR sum31 = bias1;

    uint iw, ih, idd;
    _ConvInArr.GetDimensions(iw, ih, idd);
    int groupOx = (int)groupId.x * 32;
    int groupOy = (int)groupId.y * 8;
    int groupSx = groupOx - _Pad;
    int groupSy = groupOy - _Pad;
    int lane = (int)groupThreadId.y * 16 + (int)groupThreadId.x;
    int localOx = (int)groupThreadId.x * 2;
    int localOy = (int)groupThreadId.y * 2;

    bool hasOp0 = op >= 0 && op < _OutPacks;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        for (int li = lane; li < 34 * 10; li += 64)
        {
            int tx = li % 34;
            int ty = li / 34;
            int sx = groupSx + tx;
            int sy = groupSy + ty;
            AEXIS_CONV_VECTOR v = 0.0;
            if (sx >= 0 && sy >= 0 && sx < (int)iw && sy < (int)ih)
                v = _ConvInArr[int3(sx, sy, ip0)];
            _ConvTile[li] = v;
        }

        for (int wi = lane; wi < 9 * 4; wi += 64)
        {
            _ConvWeight0[wi] = hasOp0 ? AexisReadPack4ConvWeight4(((op * _InPacks + ip0) * 9) * 4 + wi) : 0.0;
            _ConvWeight1[wi] = hasOp1 ? AexisReadPack4ConvWeight4(((op1 * _InPacks + ip0) * 9) * 4 + wi) : 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int ky = 0; ky < 3; ky++)
        {
            [unroll]
            for (int kx = 0; kx < 3; kx++)
            {
                AEXIS_CONV_VECTOR v00 = _ConvTile[(localOy + ky) * 34 + (localOx + kx)];
                AEXIS_CONV_VECTOR v01 = _ConvTile[(localOy + ky) * 34 + (localOx + kx + 1)];
                AEXIS_CONV_VECTOR v10 = _ConvTile[(localOy + ky + 1) * 34 + (localOx + kx)];
                AEXIS_CONV_VECTOR v11 = _ConvTile[(localOy + ky + 1) * 34 + (localOx + kx + 1)];

                int k = ky * 3 + kx;
                int wbase = k * 4;
                AEXIS_CONV_VECTOR w00 = _ConvWeight0[wbase + 0];
                AEXIS_CONV_VECTOR w01 = _ConvWeight0[wbase + 1];
                AEXIS_CONV_VECTOR w02 = _ConvWeight0[wbase + 2];
                AEXIS_CONV_VECTOR w03 = _ConvWeight0[wbase + 3];

                sum00.x += dot(v00, w00);
                sum00.y += dot(v00, w01);
                sum00.z += dot(v00, w02);
                sum00.w += dot(v00, w03);

                sum01.x += dot(v01, w00);
                sum01.y += dot(v01, w01);
                sum01.z += dot(v01, w02);
                sum01.w += dot(v01, w03);

                sum10.x += dot(v10, w00);
                sum10.y += dot(v10, w01);
                sum10.z += dot(v10, w02);
                sum10.w += dot(v10, w03);

                sum11.x += dot(v11, w00);
                sum11.y += dot(v11, w01);
                sum11.z += dot(v11, w02);
                sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    AEXIS_CONV_VECTOR w10 = _ConvWeight1[wbase + 0];
                    AEXIS_CONV_VECTOR w11 = _ConvWeight1[wbase + 1];
                    AEXIS_CONV_VECTOR w12 = _ConvWeight1[wbase + 2];
                    AEXIS_CONV_VECTOR w13 = _ConvWeight1[wbase + 3];

                    sum20.x += dot(v00, w10);
                    sum20.y += dot(v00, w11);
                    sum20.z += dot(v00, w12);
                    sum20.w += dot(v00, w13);

                    sum21.x += dot(v01, w10);
                    sum21.y += dot(v01, w11);
                    sum21.z += dot(v01, w12);
                    sum21.w += dot(v01, w13);

                    sum30.x += dot(v10, w10);
                    sum30.y += dot(v10, w11);
                    sum30.z += dot(v10, w12);
                    sum30.w += dot(v10, w13);

                    sum31.x += dot(v11, w10);
                    sum31.y += dot(v11, w11);
                    sum31.z += dot(v11, w12);
                    sum31.w += dot(v11, w13);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    sum00 = AexisApplyActivation(sum00);
    sum01 = AexisApplyActivation(sum01);
    sum10 = AexisApplyActivation(sum10);
    sum11 = AexisApplyActivation(sum11);
    sum00 = AexisApplyFusedConvPRelu(sum00, op);
    sum01 = AexisApplyFusedConvPRelu(sum01, op);
    sum10 = AexisApplyFusedConvPRelu(sum10, op);
    sum11 = AexisApplyFusedConvPRelu(sum11, op);
    if (hasOp1)
    {
        sum20 = AexisApplyActivation(sum20);
        sum21 = AexisApplyActivation(sum21);
        sum30 = AexisApplyActivation(sum30);
        sum31 = AexisApplyActivation(sum31);
        sum20 = AexisApplyFusedConvPRelu(sum20, op1);
        sum21 = AexisApplyFusedConvPRelu(sum21, op1);
        sum30 = AexisApplyFusedConvPRelu(sum30, op1);
        sum31 = AexisApplyFusedConvPRelu(sum31, op1);
    }

    if (active)
    {
        _ConvOutArr[int3(ox, oy, op)] = sum00;
        if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
        if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
        if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;
        if (hasOp1)
        {
            _ConvOutArr[int3(ox, oy, op1)] = sum20;
            if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
            if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
            if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
        }
    }
}

// Fixed 5x5 Pack4 path for the SESR head/tail convolution contract.  General
// convolution cannot share these assumptions (full packs, square stride-1,
// symmetric pad-2), so it remains below as the conservative fallback.
void AexisConv5x5Pack4_Impl(uint3 id, uint3 groupId, uint3 groupThreadId)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    bool active = ox < (int)ow && oy < (int)oh && op >= 0 && op < _OutPacks;
    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;

    AEXIS_CONV_VECTOR bias0 = active ? _ConvB4[op] : 0.0;
    AEXIS_CONV_VECTOR bias1 = hasOp1 ? _ConvB4[op1] : 0.0;
    AEXIS_CONV_VECTOR sum00 = bias0;
    AEXIS_CONV_VECTOR sum01 = bias0;
    AEXIS_CONV_VECTOR sum10 = bias0;
    AEXIS_CONV_VECTOR sum11 = bias0;
    AEXIS_CONV_VECTOR sum20 = bias1;
    AEXIS_CONV_VECTOR sum21 = bias1;
    AEXIS_CONV_VECTOR sum30 = bias1;
    AEXIS_CONV_VECTOR sum31 = bias1;

    uint iw, ih, idd;
    _ConvInArr.GetDimensions(iw, ih, idd);
    int groupOx = (int)groupId.x * 32;
    int groupOy = (int)groupId.y * 8;
    int groupSx = groupOx - _Pad;
    int groupSy = groupOy - _Pad;
    int lane = (int)groupThreadId.y * 16 + (int)groupThreadId.x;
    int localOx = (int)groupThreadId.x * 2;
    int localOy = (int)groupThreadId.y * 2;
    bool hasOp0 = op >= 0 && op < _OutPacks;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        for (int li = lane; li < 36 * 12; li += 64)
        {
            int tx = li % 36;
            int ty = li / 36;
            int sx = groupSx + tx;
            int sy = groupSy + ty;
            AEXIS_CONV_VECTOR v = 0.0;
            if (sx >= 0 && sy >= 0 && sx < (int)iw && sy < (int)ih)
            {
                v = _ConvInArr[int3(sx, sy, ip0)];
            }
            _Conv5Tile[li] = v;
        }

        for (int wi = lane; wi < 25 * 4; wi += 64)
        {
            _Conv5Weight0[wi] = hasOp0 ? AexisReadPack4ConvWeight4(((op * _InPacks + ip0) * 25) * 4 + wi) : 0.0;
            _Conv5Weight1[wi] = hasOp1 ? AexisReadPack4ConvWeight4(((op1 * _InPacks + ip0) * 25) * 4 + wi) : 0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int ky = 0; ky < 5; ky++)
        {
            [unroll]
            for (int kx = 0; kx < 5; kx++)
            {
                AEXIS_CONV_VECTOR v00 = _Conv5Tile[(localOy + ky) * 36 + (localOx + kx)];
                AEXIS_CONV_VECTOR v01 = _Conv5Tile[(localOy + ky) * 36 + (localOx + kx + 1)];
                AEXIS_CONV_VECTOR v10 = _Conv5Tile[(localOy + ky + 1) * 36 + (localOx + kx)];
                AEXIS_CONV_VECTOR v11 = _Conv5Tile[(localOy + ky + 1) * 36 + (localOx + kx + 1)];

                int wbase = (ky * 5 + kx) * 4;
                AEXIS_CONV_VECTOR w00 = _Conv5Weight0[wbase + 0];
                AEXIS_CONV_VECTOR w01 = _Conv5Weight0[wbase + 1];
                AEXIS_CONV_VECTOR w02 = _Conv5Weight0[wbase + 2];
                AEXIS_CONV_VECTOR w03 = _Conv5Weight0[wbase + 3];
                sum00.x += dot(v00, w00); sum00.y += dot(v00, w01); sum00.z += dot(v00, w02); sum00.w += dot(v00, w03);
                sum01.x += dot(v01, w00); sum01.y += dot(v01, w01); sum01.z += dot(v01, w02); sum01.w += dot(v01, w03);
                sum10.x += dot(v10, w00); sum10.y += dot(v10, w01); sum10.z += dot(v10, w02); sum10.w += dot(v10, w03);
                sum11.x += dot(v11, w00); sum11.y += dot(v11, w01); sum11.z += dot(v11, w02); sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    AEXIS_CONV_VECTOR w10 = _Conv5Weight1[wbase + 0];
                    AEXIS_CONV_VECTOR w11 = _Conv5Weight1[wbase + 1];
                    AEXIS_CONV_VECTOR w12 = _Conv5Weight1[wbase + 2];
                    AEXIS_CONV_VECTOR w13 = _Conv5Weight1[wbase + 3];
                    sum20.x += dot(v00, w10); sum20.y += dot(v00, w11); sum20.z += dot(v00, w12); sum20.w += dot(v00, w13);
                    sum21.x += dot(v01, w10); sum21.y += dot(v01, w11); sum21.z += dot(v01, w12); sum21.w += dot(v01, w13);
                    sum30.x += dot(v10, w10); sum30.y += dot(v10, w11); sum30.z += dot(v10, w12); sum30.w += dot(v10, w13);
                    sum31.x += dot(v11, w10); sum31.y += dot(v11, w11); sum31.z += dot(v11, w12); sum31.w += dot(v11, w13);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    sum00 = AexisApplyActivation(sum00); sum01 = AexisApplyActivation(sum01);
    sum10 = AexisApplyActivation(sum10); sum11 = AexisApplyActivation(sum11);
    sum00 = AexisApplyFusedConvPRelu(sum00, op); sum01 = AexisApplyFusedConvPRelu(sum01, op);
    sum10 = AexisApplyFusedConvPRelu(sum10, op); sum11 = AexisApplyFusedConvPRelu(sum11, op);
    if (hasOp1)
    {
        sum20 = AexisApplyActivation(sum20); sum21 = AexisApplyActivation(sum21);
        sum30 = AexisApplyActivation(sum30); sum31 = AexisApplyActivation(sum31);
        sum20 = AexisApplyFusedConvPRelu(sum20, op1); sum21 = AexisApplyFusedConvPRelu(sum21, op1);
        sum30 = AexisApplyFusedConvPRelu(sum30, op1); sum31 = AexisApplyFusedConvPRelu(sum31, op1);
    }

    if (!active) return;
    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;
    if (!hasOp1) return;
    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (ox + 1 < (int)ow) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (oy + 1 < (int)oh) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (ox + 1 < (int)ow && oy + 1 < (int)oh) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

// Raw OIHW INT8 counterpart for SESR's 5x5 head/tail. The scalar group kernel
// computes one pixel/output-pack per invocation and repeats every input sample
// for four lanes. This retains its byte-weight ABI and FP32 accumulation, but
// reuses each float4 input and vector of four channel weights across 2x2 pixels
// and two output packs. It also masks the fourth lane of SESR's RGB input.
void AexisConv5x5RawInt8Pack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    if (ox >= (int)ow || oy >= (int)oh || op < 0 || op >= (int)od)
        return;

    int op1 = op + 1;
    bool hasOp1 = op1 < (int)od;
    bool hasX1 = ox + 1 < (int)ow;
    bool hasY1 = oy + 1 < (int)oh;
    float4 bias0 = float4(_ConvB[op * 4 + 0], _ConvB[op * 4 + 1], _ConvB[op * 4 + 2], _ConvB[op * 4 + 3]);
    float4 bias1 = hasOp1
        ? float4(_ConvB[op1 * 4 + 0], _ConvB[op1 * 4 + 1], _ConvB[op1 * 4 + 2], _ConvB[op1 * 4 + 3])
        : 0.0;
    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;
    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;
    int inPacks = (_InC + 3) / 4;

    for (int ip = 0; ip < inPacks; ++ip)
    {
        int inputChannelBase = ip * 4;
        for (int ky = 0; ky < 5; ++ky)
        {
            int iy0 = oy - 2 + ky;
            int iy1 = iy0 + 1;
            bool validY0 = iy0 >= 0 && iy0 < _InH;
            bool validY1 = hasY1 && iy1 >= 0 && iy1 < _InH;
            if (!validY0 && !validY1)
                continue;

            for (int kx = 0; kx < 5; ++kx)
            {
                int ix0 = ox - 2 + kx;
                int ix1 = ix0 + 1;
                bool validX0 = ix0 >= 0 && ix0 < _InW;
                bool validX1 = hasX1 && ix1 >= 0 && ix1 < _InW;
                if (!validX0 && !validX1)
                    continue;

                float4 v00 = (validX0 && validY0) ? _ConvInArr[int3(ix0, iy0, ip)] : 0.0;
                float4 v01 = (validX1 && validY0) ? _ConvInArr[int3(ix1, iy0, ip)] : 0.0;
                float4 v10 = (validX0 && validY1) ? _ConvInArr[int3(ix0, iy1, ip)] : 0.0;
                float4 v11 = (validX1 && validY1) ? _ConvInArr[int3(ix1, iy1, ip)] : 0.0;
                if (inputChannelBase + 3 >= _InC)
                {
                    if (inputChannelBase + 0 >= _InC) { v00.x = 0.0; v01.x = 0.0; v10.x = 0.0; v11.x = 0.0; }
                    if (inputChannelBase + 1 >= _InC) { v00.y = 0.0; v01.y = 0.0; v10.y = 0.0; v11.y = 0.0; }
                    if (inputChannelBase + 2 >= _InC) { v00.z = 0.0; v01.z = 0.0; v10.z = 0.0; v11.z = 0.0; }
                    v00.w = 0.0; v01.w = 0.0; v10.w = 0.0; v11.w = 0.0;
                }
                // Keep the official QDQ graph's activation contract. Unlike the
                // scalar group kernel this is evaluated once per source float4,
                // then reused by all eight output channels below.
                v00 = float4(AexisQuantizeActivationForInt8(v00.x), AexisQuantizeActivationForInt8(v00.y), AexisQuantizeActivationForInt8(v00.z), AexisQuantizeActivationForInt8(v00.w));
                v01 = float4(AexisQuantizeActivationForInt8(v01.x), AexisQuantizeActivationForInt8(v01.y), AexisQuantizeActivationForInt8(v01.z), AexisQuantizeActivationForInt8(v01.w));
                v10 = float4(AexisQuantizeActivationForInt8(v10.x), AexisQuantizeActivationForInt8(v10.y), AexisQuantizeActivationForInt8(v10.z), AexisQuantizeActivationForInt8(v10.w));
                v11 = float4(AexisQuantizeActivationForInt8(v11.x), AexisQuantizeActivationForInt8(v11.y), AexisQuantizeActivationForInt8(v11.z), AexisQuantizeActivationForInt8(v11.w));

                int kernelIndex = ky * 5 + kx;
                float4 w00 = AexisReadRawConvWeight4(op * 4 + 0, inputChannelBase, kernelIndex);
                float4 w01 = AexisReadRawConvWeight4(op * 4 + 1, inputChannelBase, kernelIndex);
                float4 w02 = AexisReadRawConvWeight4(op * 4 + 2, inputChannelBase, kernelIndex);
                float4 w03 = AexisReadRawConvWeight4(op * 4 + 3, inputChannelBase, kernelIndex);
                sum00.x += dot(v00, w00); sum00.y += dot(v00, w01); sum00.z += dot(v00, w02); sum00.w += dot(v00, w03);
                sum01.x += dot(v01, w00); sum01.y += dot(v01, w01); sum01.z += dot(v01, w02); sum01.w += dot(v01, w03);
                sum10.x += dot(v10, w00); sum10.y += dot(v10, w01); sum10.z += dot(v10, w02); sum10.w += dot(v10, w03);
                sum11.x += dot(v11, w00); sum11.y += dot(v11, w01); sum11.z += dot(v11, w02); sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    float4 w10 = AexisReadRawConvWeight4(op1 * 4 + 0, inputChannelBase, kernelIndex);
                    float4 w11 = AexisReadRawConvWeight4(op1 * 4 + 1, inputChannelBase, kernelIndex);
                    float4 w12 = AexisReadRawConvWeight4(op1 * 4 + 2, inputChannelBase, kernelIndex);
                    float4 w13 = AexisReadRawConvWeight4(op1 * 4 + 3, inputChannelBase, kernelIndex);
                    sum20.x += dot(v00, w10); sum20.y += dot(v00, w11); sum20.z += dot(v00, w12); sum20.w += dot(v00, w13);
                    sum21.x += dot(v01, w10); sum21.y += dot(v01, w11); sum21.z += dot(v01, w12); sum21.w += dot(v01, w13);
                    sum30.x += dot(v10, w10); sum30.y += dot(v10, w11); sum30.z += dot(v10, w12); sum30.w += dot(v10, w13);
                    sum31.x += dot(v11, w10); sum31.y += dot(v11, w11); sum31.z += dot(v11, w12); sum31.w += dot(v11, w13);
                }
            }
        }
    }

    sum00 = AexisApplyActivation(sum00);
    if (hasX1) sum01 = AexisApplyActivation(sum01);
    if (hasY1) sum10 = AexisApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = AexisApplyActivation(sum11);
    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;

    if (!hasOp1)
        return;

    sum20 = AexisApplyActivation(sum20);
    if (hasX1) sum21 = AexisApplyActivation(sum21);
    if (hasY1) sum30 = AexisApplyActivation(sum30);
    if (hasX1 && hasY1) sum31 = AexisApplyActivation(sum31);
    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

void AexisConvPack4General_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    if (ox >= (int)ow || oy >= (int)oh || op < 0 || op >= _OutPacks)
        return;

    int op1 = op + 1;
    bool hasOp1 = op1 < _OutPacks;
    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;
    int kernelArea = _KernelWVar * _KernelHVar;
    float4 bias0 = _ConvB4[op];
    float4 bias1 = hasOp1 ? _ConvB4[op1] : 0.0;
    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;
    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int inputChannelBase = ip0 * 4;
        int weightBase0 = ((op * _InPacks + ip0) * kernelArea) * 4;
        int weightBase1 = ((op1 * _InPacks + ip0) * kernelArea) * 4;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int iy0 = oy * _StrideHVar - _PadTopVar + ky * _DilationHVar;
            int iy1 = iy0 + _StrideHVar;
            bool validY0 = iy0 >= 0 && iy0 < _InH;
            bool validY1 = hasY1 && iy1 >= 0 && iy1 < _InH;
            if (!validY0 && !validY1)
                continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int ix0 = ox * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                int ix1 = ix0 + _StrideWVar;
                bool validX0 = ix0 >= 0 && ix0 < _InW;
                bool validX1 = hasX1 && ix1 >= 0 && ix1 < _InW;
                if (!validX0 && !validX1)
                    continue;

                float4 v00 = (validX0 && validY0) ? _ConvInArr[int3(ix0, iy0, ip0)] : 0.0;
                float4 v01 = (validX1 && validY0) ? _ConvInArr[int3(ix1, iy0, ip0)] : 0.0;
                float4 v10 = (validX0 && validY1) ? _ConvInArr[int3(ix0, iy1, ip0)] : 0.0;
                float4 v11 = (validX1 && validY1) ? _ConvInArr[int3(ix1, iy1, ip0)] : 0.0;

                int kernelIndex = ky * _KernelWVar + kx;
                int baseIndex0 = weightBase0 + kernelIndex * 4;
                // Keep the ordinary floating point path on its established
                // compact O4I4K reader.  The raw-INT8 expansion has a different
                // channel ABI and must be selected only by an INT8 dispatch.
                // Routing normal FP16/FP32 5x5 SESR layers through that helper
                // made their head/tail tensors collapse to zero on D3D11.
                float4 w00 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 0, op * 4 + 0, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 0);
                float4 w01 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 1, op * 4 + 1, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 1);
                float4 w02 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 2, op * 4 + 2, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 2);
                float4 w03 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 3, op * 4 + 3, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 3);
                sum00.x += dot(v00, w00);
                sum00.y += dot(v00, w01);
                sum00.z += dot(v00, w02);
                sum00.w += dot(v00, w03);
                sum01.x += dot(v01, w00);
                sum01.y += dot(v01, w01);
                sum01.z += dot(v01, w02);
                sum01.w += dot(v01, w03);
                sum10.x += dot(v10, w00);
                sum10.y += dot(v10, w01);
                sum10.z += dot(v10, w02);
                sum10.w += dot(v10, w03);
                sum11.x += dot(v11, w00);
                sum11.y += dot(v11, w01);
                sum11.z += dot(v11, w02);
                sum11.w += dot(v11, w03);

                if (hasOp1)
                {
                    int baseIndex1 = weightBase1 + kernelIndex * 4;
                    float4 w10 = _UseInt8ConvWeights != 0
                        ? AexisReadGenericConvWeight4(baseIndex1 + 0, op1 * 4 + 0, inputChannelBase, kernelIndex, kernelArea)
                        : AexisReadConvWeight4(baseIndex1 + 0);
                    float4 w11 = _UseInt8ConvWeights != 0
                        ? AexisReadGenericConvWeight4(baseIndex1 + 1, op1 * 4 + 1, inputChannelBase, kernelIndex, kernelArea)
                        : AexisReadConvWeight4(baseIndex1 + 1);
                    float4 w12 = _UseInt8ConvWeights != 0
                        ? AexisReadGenericConvWeight4(baseIndex1 + 2, op1 * 4 + 2, inputChannelBase, kernelIndex, kernelArea)
                        : AexisReadConvWeight4(baseIndex1 + 2);
                    float4 w13 = _UseInt8ConvWeights != 0
                        ? AexisReadGenericConvWeight4(baseIndex1 + 3, op1 * 4 + 3, inputChannelBase, kernelIndex, kernelArea)
                        : AexisReadConvWeight4(baseIndex1 + 3);
                    sum20.x += dot(v00, w10);
                    sum20.y += dot(v00, w11);
                    sum20.z += dot(v00, w12);
                    sum20.w += dot(v00, w13);
                    sum21.x += dot(v01, w10);
                    sum21.y += dot(v01, w11);
                    sum21.z += dot(v01, w12);
                    sum21.w += dot(v01, w13);
                    sum30.x += dot(v10, w10);
                    sum30.y += dot(v10, w11);
                    sum30.z += dot(v10, w12);
                    sum30.w += dot(v10, w13);
                    sum31.x += dot(v11, w10);
                    sum31.y += dot(v11, w11);
                    sum31.z += dot(v11, w12);
                    sum31.w += dot(v11, w13);
                }
            }
        }
    }

    sum00 = AexisApplyActivation(sum00);
    if (hasX1) sum01 = AexisApplyActivation(sum01);
    if (hasY1) sum10 = AexisApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = AexisApplyActivation(sum11);
    sum00 = AexisMaskConvOutputTail(sum00, op);
    if (hasX1) sum01 = AexisMaskConvOutputTail(sum01, op);
    if (hasY1) sum10 = AexisMaskConvOutputTail(sum10, op);
    if (hasX1 && hasY1) sum11 = AexisMaskConvOutputTail(sum11, op);

    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;

    if (!hasOp1)
        return;

    sum20 = AexisApplyActivation(sum20);
    if (hasX1) sum21 = AexisApplyActivation(sum21);
    if (hasY1) sum30 = AexisApplyActivation(sum30);
    if (hasX1 && hasY1) sum31 = AexisApplyActivation(sum31);
    sum20 = AexisMaskConvOutputTail(sum20, op1);
    if (hasX1) sum21 = AexisMaskConvOutputTail(sum21, op1);
    if (hasY1) sum30 = AexisMaskConvOutputTail(sum30, op1);
    if (hasX1 && hasY1) sum31 = AexisMaskConvOutputTail(sum31, op1);

    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

// Flat 1-output-per-thread variant of the general Pack4 convolution. One thread
// computes exactly one output texel (one spatial pixel x, one channel pack). It
// intentionally drops the 2x2 output tile: for narrow 1-row vocoder stages the
// tiled kernel launches an 8x8 workgroup whose y-dimension threads immediately
// bail (height==1), wasting ~7/8 of slots across all 12 SMs. Numerically this
// mirrors only the sum00 path of AexisConvPack4General_Impl for (ox,oy,op), so
// results are identical to the tiled kernel for the same output coordinate.
void AexisConvPack4GeneralFlat_Impl(int ox, int oy, int op)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (ox < 0 || ox >= (int)ow || oy < 0 || oy >= (int)oh || op < 0 || op >= _OutPacks)
        return;

    int kernelArea = _KernelWVar * _KernelHVar;
    float4 sum = _ConvB4[op];

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int inputChannelBase = ip0 * 4;
        int weightBase0 = ((op * _InPacks + ip0) * kernelArea) * 4;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int iy0 = oy * _StrideHVar - _PadTopVar + ky * _DilationHVar;
            bool validY0 = iy0 >= 0 && iy0 < _InH;
            if (!validY0)
                continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int ix0 = ox * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                bool validX0 = ix0 >= 0 && ix0 < _InW;
                if (!validX0)
                    continue;

                float4 v00 = _ConvInArr[int3(ix0, iy0, ip0)];

                int kernelIndex = ky * _KernelWVar + kx;
                int baseIndex0 = weightBase0 + kernelIndex * 4;
                float4 w00 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 0, op * 4 + 0, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 0);
                float4 w01 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 1, op * 4 + 1, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 1);
                float4 w02 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 2, op * 4 + 2, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 2);
                float4 w03 = _UseInt8ConvWeights != 0
                    ? AexisReadGenericConvWeight4(baseIndex0 + 3, op * 4 + 3, inputChannelBase, kernelIndex, kernelArea)
                    : AexisReadConvWeight4(baseIndex0 + 3);
                sum.x += dot(v00, w00);
                sum.y += dot(v00, w01);
                sum.z += dot(v00, w02);
                sum.w += dot(v00, w03);
            }
        }
    }

    sum = AexisApplyActivation(sum);
    sum = AexisMaskConvOutputTail(sum, op);
    _ConvOutArr[int3(ox, oy, op)] = sum;
}

// 1D register-tiled conv: each thread computes TILE_X=4 output pixels along x,
// caching input texels in registers to reuse across adjacent outputs. This raises
// arithmetic intensity from ~0.25 to ~0.66 FLOP/byte for narrow 1-row vocoder stages.
// Only valid for logical height==1 (oy==0), stride==1, dilationH==1.
#define AEXIS_TILE_X 4
float4 AexisReadTileXConvInput(int ix, int ip0)
{
    if (ix < 0 || ix >= _InW) return 0.0;
    return _InTileW > 0
        ? _ConvInArr[int3(ix % _InTileW, ix / _InTileW, ip0)]
        : _ConvInArr[int3(ix, 0, ip0)];
}

void AexisConvPack4GeneralTileX_Impl(int ox, int oy, int op)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int logicalOutW = _OutTileW > 0 ? _OutW : (int)ow;
    if (ox >= logicalOutW || op < 0 || op >= _OutPacks) return;

    int kernelW = _KernelWVar;
    int kernelH = _KernelHVar;
    int padLeft = _PadLeftVar;

    // Accumulators for TILE_X output pixels
    float4 sum0 = _ConvB4[op];
    float4 sum1 = _ConvB4[op];
    float4 sum2 = _ConvB4[op];
    float4 sum3 = _ConvB4[op];

    // Input cache: up to TILE_X + kernelW - 1 values
    float4 inCache[16];

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        // output[ox + t] uses input[ox + t - padLeft + kx * dilationW]
        int inStart = ox - padLeft;
        if (_DilationWVar == 1)
        {
            int cacheCount = AEXIS_TILE_X + kernelW - 1;
            for (int c = 0; c < cacheCount; c++)
                inCache[c] = AexisReadTileXConvInput(inStart + c, ip0);
        }

        // Compute dot products
        int weightBase = ((op * _InPacks + ip0) * kernelW) * 4;
        for (int kx = 0; kx < kernelW; kx++)
        {
            int wBase = weightBase + kx * 4;
            float4 w00 = _UseInt8ConvWeights != 0
                ? AexisReadGenericConvWeight4(wBase + 0, op * 4 + 0, ip0 * 4, kx, kernelW)
                : AexisReadConvWeight4(wBase + 0);
            float4 w01 = _UseInt8ConvWeights != 0
                ? AexisReadGenericConvWeight4(wBase + 1, op * 4 + 1, ip0 * 4, kx, kernelW)
                : AexisReadConvWeight4(wBase + 1);
            float4 w02 = _UseInt8ConvWeights != 0
                ? AexisReadGenericConvWeight4(wBase + 2, op * 4 + 2, ip0 * 4, kx, kernelW)
                : AexisReadConvWeight4(wBase + 2);
            float4 w03 = _UseInt8ConvWeights != 0
                ? AexisReadGenericConvWeight4(wBase + 3, op * 4 + 3, ip0 * 4, kx, kernelW)
                : AexisReadConvWeight4(wBase + 3);

            float4 v0, v1, v2, v3;
            if (_DilationWVar == 1)
            {
                v0 = inCache[0 + kx];
                v1 = inCache[1 + kx];
                v2 = inCache[2 + kx];
                v3 = inCache[3 + kx];
            }
            else
            {
                int ix = inStart + kx * _DilationWVar;
                v0 = AexisReadTileXConvInput(ix + 0, ip0);
                v1 = AexisReadTileXConvInput(ix + 1, ip0);
                v2 = AexisReadTileXConvInput(ix + 2, ip0);
                v3 = AexisReadTileXConvInput(ix + 3, ip0);
            }

            sum0.x += dot(v0, w00); sum0.y += dot(v0, w01); sum0.z += dot(v0, w02); sum0.w += dot(v0, w03);
            sum1.x += dot(v1, w00); sum1.y += dot(v1, w01); sum1.z += dot(v1, w02); sum1.w += dot(v1, w03);
            sum2.x += dot(v2, w00); sum2.y += dot(v2, w01); sum2.z += dot(v2, w02); sum2.w += dot(v2, w03);
            sum3.x += dot(v3, w00); sum3.y += dot(v3, w01); sum3.z += dot(v3, w02); sum3.w += dot(v3, w03);
        }
    }

    // Apply activation and write outputs
    sum0 = AexisApplyActivation(sum0);
    sum1 = AexisApplyActivation(sum1);
    sum2 = AexisApplyActivation(sum2);
    sum3 = AexisApplyActivation(sum3);
    sum0 = AexisMaskConvOutputTail(sum0, op);
    sum1 = AexisMaskConvOutputTail(sum1, op);
    sum2 = AexisMaskConvOutputTail(sum2, op);
    sum3 = AexisMaskConvOutputTail(sum3, op);

    if (_OutTileW > 0)
    {
        if (ox < logicalOutW) _ConvOutArr[int3(ox % _OutTileW, ox / _OutTileW, op)] = sum0;
        if (ox + 1 < logicalOutW) _ConvOutArr[int3((ox + 1) % _OutTileW, (ox + 1) / _OutTileW, op)] = sum1;
        if (ox + 2 < logicalOutW) _ConvOutArr[int3((ox + 2) % _OutTileW, (ox + 2) / _OutTileW, op)] = sum2;
        if (ox + 3 < logicalOutW) _ConvOutArr[int3((ox + 3) % _OutTileW, (ox + 3) / _OutTileW, op)] = sum3;
    }
    else
    {
        if (ox < logicalOutW) _ConvOutArr[int3(ox, 0, op)] = sum0;
        if (ox + 1 < logicalOutW) _ConvOutArr[int3(ox + 1, 0, op)] = sum1;
        if (ox + 2 < logicalOutW) _ConvOutArr[int3(ox + 2, 0, op)] = sum2;
        if (ox + 3 < logicalOutW) _ConvOutArr[int3(ox + 3, 0, op)] = sum3;
    }
}

void AexisDeconvolutionPack4General_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int op = (int)id.z;
    if (op < 0 || op >= _OutPacks) return;

    int kernelArea = _KernelWVar * _KernelHVar;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = _ConvB4[op];

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int weightBase = ((op * _InPacks + ip0) * kernelArea) * 4;

        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int iyNumerator = borderedY - ky * _DilationHVar;
            if (iyNumerator < 0) continue;
            if ((iyNumerator % _StrideHVar) != 0) continue;
            int iy = iyNumerator / _StrideHVar;
            if (iy < 0 || iy >= _InH) continue;

            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int ixNumerator = borderedX - kx * _DilationWVar;
                if (ixNumerator < 0) continue;
                if ((ixNumerator % _StrideWVar) != 0) continue;
                int ix = ixNumerator / _StrideWVar;
                if (ix < 0 || ix >= _InW) continue;

                float4 inV = _ConvInArr[int3(ix, iy, ip0)];
                int baseIndex = weightBase + (ky * _KernelWVar + kx) * 4;
                float4 w0 = AexisReadConvWeight4(baseIndex + 0);
                float4 w1 = AexisReadConvWeight4(baseIndex + 1);
                float4 w2 = AexisReadConvWeight4(baseIndex + 2);
                float4 w3 = AexisReadConvWeight4(baseIndex + 3);
                sum.x += dot(inV, w0);
                sum.y += dot(inV, w1);
                sum.z += dot(inV, w2);
                sum.w += dot(inV, w3);
            }
        }
    }

    sum = AexisApplyActivation(sum);
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

// Wide deconv tiling: each thread computes TILE_X=8 output pixels along x.
// The point of this kernel over the 8x8 3D default is the 1D dispatch: for the
// height==1 vocoder stages the 3D kernel leaves 7/8 of its y-threads idle.
// Input texels are read straight from _ConvInArr (the texture cache handles the
// reuse); an earlier register cache was capped at 4 entries, which silently
// dropped taps for stride=2/kernel=8 stages where 6 distinct inputs feed a tile.
#define AEXIS_DECONV_TILE_X 8
void AexisDeconvolutionPack4GeneralTileX_Impl(int baseOx, int oy, int op)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int logicalOutW = _OutTileW > 0 ? _OutW : (int)ow;
    if (baseOx >= logicalOutW || op < 0 || op >= _OutPacks) return;

    int kernelW = _KernelWVar;
    int strideW = _StrideWVar;
    int padLeft = _PadLeftVar;

    // Accumulators for TILE_X output pixels
    float4 sums[AEXIS_DECONV_TILE_X];
    for (int t = 0; t < AEXIS_DECONV_TILE_X; t++)
        sums[t] = _ConvB4[op];

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        int weightBase = ((op * _InPacks + ip0) * kernelW) * 4;
        // Share each tap's weights across the output pixels in this tile. The
        // kx loop remains in ascending order for every sums[t] accumulator.
        for (int kx = 0; kx < kernelW; kx++)
        {
            int baseIndex = weightBase + kx * 4;
            float4 w0 = AexisReadConvWeight4(baseIndex + 0);
            float4 w1 = AexisReadConvWeight4(baseIndex + 1);
            float4 w2 = AexisReadConvWeight4(baseIndex + 2);
            float4 w3 = AexisReadConvWeight4(baseIndex + 3);
            for (int t = 0; t < AEXIS_DECONV_TILE_X; t++)
            {
                int ox = baseOx + t;
                int borderedX = ox + padLeft;
                int ixNumerator = borderedX - kx;
                if (ixNumerator < 0) continue;
                if ((ixNumerator % strideW) != 0) continue;
                int ix = ixNumerator / strideW;
                if (ix >= _InW) continue;

                float4 inV = _InTileW > 0
                    ? _ConvInArr[int3(ix % _InTileW, ix / _InTileW, ip0)]
                    : _ConvInArr[int3(ix, oy, ip0)];
                sums[t].x += dot(inV, w0);
                sums[t].y += dot(inV, w1);
                sums[t].z += dot(inV, w2);
                sums[t].w += dot(inV, w3);
            }
        }
    }

    // Apply activation and write outputs
    // 2D layout: when _OutTileW > 0, write to (ox % _OutTileW, ox / _OutTileW) instead of (ox, oy)
    // This allows outputs wider than maxTextureSize (16384) by wrapping into multiple rows.
    for (int t = 0; t < AEXIS_DECONV_TILE_X; t++)
    {
        int ox = baseOx + t;
        if (ox >= logicalOutW) break;
        sums[t] = AexisApplyActivation(sums[t]);
        if (_OutTileW > 0)
        {
            int wx = ox % _OutTileW;
            int wy = ox / _OutTileW;
            _ConvOutArr[int3(wx, wy, op)] = sums[t];
        }
        else
        {
            _ConvOutArr[int3(ox, oy, op)] = sums[t];
        }
    }
}

// General 2D Pack4 convolution uses immutable scalar ncnn weights. Activations and
// results remain Texture2DArray-backed; scalar reads are necessary because group
// boundaries and channel tails are not required to align to a float4 pack.
void AexisConv2dGroupPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int op = (int)id.z;
    int group = max(1, _ConvGroup);
    int inchG = _InC / group;
    int outchG = _OutC / group;
    int kernelArea = _KernelWVar * _KernelHVar;
    float4 sum = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int oc = op * 4 + lane;
        if (oc >= _OutC)
            continue;

        int g = min(group - 1, oc / outchG);
        float value = _ConvB[oc];
        int weightBase = (oc * inchG) * kernelArea;
        for (int icLocal = 0; icLocal < inchG; icLocal++)
        {
            int ic = g * inchG + icLocal;
            int kernelBase = weightBase + icLocal * kernelArea;
            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int iy = (int)id.y * _StrideHVar - _PadTopVar + ky * _DilationHVar;
                if (iy < 0 || iy >= _InH)
                    continue;
                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int ix = (int)id.x * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                    if (ix < 0 || ix >= _InW)
                        continue;
                    value += AexisQuantizeActivationForInt8(AexisReadPack4Channel(_ConvInArr, ix, iy, ic)) * AexisReadRawConvWeight(kernelBase + ky * _KernelWVar + kx, oc);
                }
            }
        }
        AexisWriteLane(sum, lane, value);
    }

    sum = AexisApplyActivation(sum);
    [unroll]
    for (int tailLane = 0; tailLane < 4; tailLane++)
    {
        if (op * 4 + tailLane >= _OutC)
            AexisWriteLane(sum, tailLane, 0.0);
    }
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

void AexisDeconvolution2dGroupPack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh || id.z >= od)
        return;

    int op = (int)id.z;
    int group = max(1, _ConvGroup);
    int inchG = _InC / group;
    int outchG = _OutC / group;
    int kernelArea = _KernelWVar * _KernelHVar;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = 0.0;

    [unroll]
    for (int lane = 0; lane < 4; lane++)
    {
        int oc = op * 4 + lane;
        if (oc >= _OutC)
            continue;

        int g = min(group - 1, oc / outchG);
        float value = _ConvB[oc];
        int weightBase = (oc * inchG) * kernelArea;
        for (int icLocal = 0; icLocal < inchG; icLocal++)
        {
            int ic = g * inchG + icLocal;
            int kernelBase = weightBase + icLocal * kernelArea;
            for (int ky = 0; ky < _KernelHVar; ky++)
            {
                int iyNumerator = borderedY - ky * _DilationHVar;
                if (iyNumerator < 0 || (iyNumerator % _StrideHVar) != 0)
                    continue;
                int iy = iyNumerator / _StrideHVar;
                if (iy < 0 || iy >= _InH)
                    continue;
                for (int kx = 0; kx < _KernelWVar; kx++)
                {
                    int ixNumerator = borderedX - kx * _DilationWVar;
                    if (ixNumerator < 0 || (ixNumerator % _StrideWVar) != 0)
                        continue;
                    int ix = ixNumerator / _StrideWVar;
                    if (ix < 0 || ix >= _InW)
                        continue;
                    value += AexisQuantizeActivationForInt8(AexisReadPack4Channel(_ConvInArr, ix, iy, ic)) * AexisReadRawConvWeight(kernelBase + ky * _KernelWVar + kx, oc);
                }
            }
        }
        AexisWriteLane(sum, lane, value);
    }

    sum = AexisApplyActivation(sum);
    [unroll]
    for (int tailLane = 0; tailLane < 4; tailLane++)
    {
        if (op * 4 + tailLane >= _OutC)
            AexisWriteLane(sum, tailLane, 0.0);
    }
    _ConvOutArr[int3((int)id.x, (int)id.y, op)] = sum;
}

void AexisDeconvolutionDepthWisePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    if (id.x >= ow || id.y >= oh) return;
    int pack = (int)id.z;
    if (pack < 0 || pack >= _OutPacks) return;

    int group = max(1, _ConvGroup);
    int outch_g = max(1, _OutC / group);
    bool isOneToOneDepthWise = _OutC == group && _InC == group;
    int borderedX = (int)id.x + _PadLeftVar;
    int borderedY = (int)id.y + _PadTopVar;
    float4 sum = _DwConvB4[pack];

    for (int ky = 0; ky < _KernelHVar; ky++)
    {
        int iyNumerator = borderedY - ky * _DilationHVar;
        if (iyNumerator < 0)
            continue;
        if ((iyNumerator % _StrideHVar) != 0)
            continue;
        int iy = iyNumerator / _StrideHVar;
        if (iy < 0 || iy >= _InH)
            continue;

        for (int kx = 0; kx < _KernelWVar; kx++)
        {
            int ixNumerator = borderedX - kx * _DilationWVar;
            if (ixNumerator < 0)
                continue;
            if ((ixNumerator % _StrideWVar) != 0)
                continue;
            int ix = ixNumerator / _StrideWVar;
            if (ix < 0 || ix >= _InW)
                continue;

            float4 w = _DwConvW4[(pack * _KernelHVar + ky) * _KernelWVar + kx];
            if (isOneToOneDepthWise)
            {
                sum += _ConvInArr[int3(ix, iy, pack)] * w;
                continue;
            }

            [unroll]
            for (int lane = 0; lane < 4; lane++)
            {
                int oc = pack * 4 + lane;
                if (oc < 0 || oc >= _OutC)
                    continue;

                int g = min(group - 1, oc / outch_g);
                float value = AexisReadLane(sum, lane) + AexisReadPack4Channel(_ConvInArr, ix, iy, g) * AexisReadLane(w, lane);
                AexisWriteLane(sum, lane, value);
            }
        }
    }

    sum = AexisApplyActivation(sum);
    _ConvOutArr[int3((int)id.x, (int)id.y, pack)] = sum;
}

void AexisConvDepthWisePack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);
    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int pack = (int)id.z;
    if (ox >= (int)ow || oy >= (int)oh || pack < 0 || pack >= _OutPacks) return;

    bool hasX1 = (ox + 1) < (int)ow;
    bool hasY1 = (oy + 1) < (int)oh;
    float4 bias = _DwConvB4[pack];
    float4 sum00 = bias;
    float4 sum01 = bias;
    float4 sum10 = bias;
    float4 sum11 = bias;
    int group = max(1, _ConvGroup);
    bool isOneToOneDepthWise = _OutC == group;
    int outch_g = max(1, _OutC / group);

    for (int ky = 0; ky < _KernelHVar; ky++)
    {
        int iy0 = oy * _StrideHVar - _PadTopVar + ky * _DilationHVar;
        int iy1 = iy0 + _StrideHVar;
        bool validY0 = iy0 >= 0 && iy0 < _InH;
        bool validY1 = hasY1 && iy1 >= 0 && iy1 < _InH;
        if (!validY0 && !validY1) continue;

        for (int kx = 0; kx < _KernelWVar; kx++)
        {
            int ix0 = ox * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
            int ix1 = ix0 + _StrideWVar;
            bool validX0 = ix0 >= 0 && ix0 < _InW;
            bool validX1 = hasX1 && ix1 >= 0 && ix1 < _InW;
            if (!validX0 && !validX1) continue;

            float4 w = AexisReadDepthWiseConvWeight4((pack * _KernelHVar + ky) * _KernelWVar + kx);
            if (isOneToOneDepthWise)
            {
                if (validX0 && validY0) sum00 += _ConvInArr[int3(ix0, iy0, pack)] * w;
                if (validX1 && validY0) sum01 += _ConvInArr[int3(ix1, iy0, pack)] * w;
                if (validX0 && validY1) sum10 += _ConvInArr[int3(ix0, iy1, pack)] * w;
                if (validX1 && validY1) sum11 += _ConvInArr[int3(ix1, iy1, pack)] * w;
                continue;
            }

            [unroll]
            for (int lane = 0; lane < 4; lane++)
            {
                int oc = pack * 4 + lane;
                if (oc < 0 || oc >= _OutC)
                    continue;

                int g = min(group - 1, oc / outch_g);
                float weightValue = AexisReadLane(w, lane);
                if (validX0 && validY0)
                    AexisWriteLane(sum00, lane, AexisReadLane(sum00, lane) + AexisReadPack4Channel(_ConvInArr, ix0, iy0, g) * weightValue);
                if (validX1 && validY0)
                    AexisWriteLane(sum01, lane, AexisReadLane(sum01, lane) + AexisReadPack4Channel(_ConvInArr, ix1, iy0, g) * weightValue);
                if (validX0 && validY1)
                    AexisWriteLane(sum10, lane, AexisReadLane(sum10, lane) + AexisReadPack4Channel(_ConvInArr, ix0, iy1, g) * weightValue);
                if (validX1 && validY1)
                    AexisWriteLane(sum11, lane, AexisReadLane(sum11, lane) + AexisReadPack4Channel(_ConvInArr, ix1, iy1, g) * weightValue);
            }
        }
    }

    sum00 = AexisApplyActivation(sum00);
    if (hasX1) sum01 = AexisApplyActivation(sum01);
    if (hasY1) sum10 = AexisApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = AexisApplyActivation(sum11);

    _ConvOutArr[int3(ox, oy, pack)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, pack)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, pack)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, pack)] = sum11;
}

void AexisWinograd23TransformInputPack4_Impl(uint3 id)
{
    int gx = (int)id.x;
    int gy = (int)id.y;
    int gz = (int)id.z;
    if (gx >= _WinoBlockX || gy >= _WinoBlockY || gz >= _InPacks)
        return;

    // ncnn winograd23 samples from the padded input tensor.
    // Repro path keeps the source unpadded, so emulate pad=1 here.
    int sx = gx * 2 - 1;
    int sy = gy * 2 - 1;
    int cstep = _WinoInCstep;

    // Every coordinate must be bounds checked.  The old fast path only checked
    // the positive edge for most samples, so the first/last tile could read a
    // negative row/column on D3D11 and contaminate the residual branch.
    float4 v00 = AexisWinoReadInput(sx + 0, sy + 0, gz);
    float4 v01 = AexisWinoReadInput(sx + 1, sy + 0, gz);
    float4 v02 = AexisWinoReadInput(sx + 2, sy + 0, gz);
    float4 v03 = AexisWinoReadInput(sx + 3, sy + 0, gz);
    float4 v10 = AexisWinoReadInput(sx + 0, sy + 1, gz);
    float4 v11 = AexisWinoReadInput(sx + 1, sy + 1, gz);
    float4 v12 = AexisWinoReadInput(sx + 2, sy + 1, gz);
    float4 v13 = AexisWinoReadInput(sx + 3, sy + 1, gz);
    float4 v20 = AexisWinoReadInput(sx + 0, sy + 2, gz);
    float4 v21 = AexisWinoReadInput(sx + 1, sy + 2, gz);
    float4 v22 = AexisWinoReadInput(sx + 2, sy + 2, gz);
    float4 v23 = AexisWinoReadInput(sx + 3, sy + 2, gz);
    float4 v30 = AexisWinoReadInput(sx + 0, sy + 3, gz);
    float4 v31 = AexisWinoReadInput(sx + 1, sy + 3, gz);
    float4 v32 = AexisWinoReadInput(sx + 2, sy + 3, gz);
    float4 v33 = AexisWinoReadInput(sx + 3, sy + 3, gz);

    float4 m00 = v00 - v02;
    float4 m01 = v10 - v12;
    float4 m02 = v20 - v22;
    float4 m03 = v30 - v32;

    float4 m10 = v02 + v01;
    float4 m11 = v12 + v11;
    float4 m12 = v22 + v21;
    float4 m13 = v32 + v31;

    float4 m20 = v02 - v01;
    float4 m21 = v12 - v11;
    float4 m22 = v22 - v21;
    float4 m23 = v32 - v31;

    float4 m30 = v03 - v01;
    float4 m31 = v13 - v11;
    float4 m32 = v23 - v21;
    float4 m33 = v33 - v31;

    v00 = m00 - m02;
    v10 = m10 - m12;
    v20 = m20 - m22;
    v30 = m30 - m32;

    v01 = m02 + m01;
    v11 = m12 + m11;
    v21 = m22 + m21;
    v31 = m32 + m31;

    v02 = m02 - m01;
    v12 = m12 - m11;
    v22 = m22 - m21;
    v32 = m32 - m31;

    v03 = m03 - m01;
    v13 = m13 - m11;
    v23 = m23 - m21;
    v33 = m33 - m31;

    int vTmOffset = gz * cstep + gy * _WinoBlockX + gx;
    int packC = _InPacks;
    _WinoBottomTm[vTmOffset + 0 * cstep * packC] = v00;
    _WinoBottomTm[vTmOffset + 1 * cstep * packC] = v01;
    _WinoBottomTm[vTmOffset + 2 * cstep * packC] = v02;
    _WinoBottomTm[vTmOffset + 3 * cstep * packC] = v03;
    _WinoBottomTm[vTmOffset + 4 * cstep * packC] = v10;
    _WinoBottomTm[vTmOffset + 5 * cstep * packC] = v11;
    _WinoBottomTm[vTmOffset + 6 * cstep * packC] = v12;
    _WinoBottomTm[vTmOffset + 7 * cstep * packC] = v13;
    _WinoBottomTm[vTmOffset + 8 * cstep * packC] = v20;
    _WinoBottomTm[vTmOffset + 9 * cstep * packC] = v21;
    _WinoBottomTm[vTmOffset + 10 * cstep * packC] = v22;
    _WinoBottomTm[vTmOffset + 11 * cstep * packC] = v23;
    _WinoBottomTm[vTmOffset + 12 * cstep * packC] = v30;
    _WinoBottomTm[vTmOffset + 13 * cstep * packC] = v31;
    _WinoBottomTm[vTmOffset + 14 * cstep * packC] = v32;
    _WinoBottomTm[vTmOffset + 15 * cstep * packC] = v33;
}

void AexisWinograd23GemmPack4_Impl(uint3 id)
{
    int gx = (int)id.x * 4;
    int gy = (int)id.y;
    int gz = (int)id.z;
    int tiles = _WinoInCstep;
    int outc = _OutPacks;
    if (gx >= tiles || gy >= outc || gz >= 16)
        return;

    float4 sum0 = 0.0;
    float4 sum1 = 0.0;
    float4 sum2 = 0.0;
    float4 sum3 = 0.0;

    // bottom_tm layout is [k][in_pack][tile], matching ncnn's
    // gz * c * cstep + gx base address in convolution_pack4_3x3s1d1_winograd_gemm.
    int vOffset = gz * _InPacks * tiles + gx;
    int wBase = (gz * outc + gy) * _InPacks * 4;

    for (int z = 0; z < _InPacks; z++)
    {
        int wOffset = wBase + z * 4;
        float4 v0 = _WinoBottomTm[vOffset + 0];
        float4 v1 = _WinoBottomTm[vOffset + 1];
        float4 v2 = _WinoBottomTm[vOffset + 2];
        float4 v3 = _WinoBottomTm[vOffset + 3];

        float4 k0 = _WinoWeightTm[wOffset + 0];
        float4 k1 = _WinoWeightTm[wOffset + 1];
        float4 k2 = _WinoWeightTm[wOffset + 2];
        float4 k3 = _WinoWeightTm[wOffset + 3];

        sum0.x += dot(v0, k0); sum0.y += dot(v0, k1); sum0.z += dot(v0, k2); sum0.w += dot(v0, k3);
        sum1.x += dot(v1, k0); sum1.y += dot(v1, k1); sum1.z += dot(v1, k2); sum1.w += dot(v1, k3);
        sum2.x += dot(v2, k0); sum2.y += dot(v2, k1); sum2.z += dot(v2, k2); sum2.w += dot(v2, k3);
        sum3.x += dot(v3, k0); sum3.y += dot(v3, k1); sum3.z += dot(v3, k2); sum3.w += dot(v3, k3);

        vOffset += tiles;
    }

    int base = gz * outc * _WinoOutCstep + gy * _WinoOutCstep + gx;
    _WinoTopTm[base + 0] = sum0;
    if (gx + 1 < tiles) _WinoTopTm[base + 1] = sum1;
    if (gx + 2 < tiles) _WinoTopTm[base + 2] = sum2;
    if (gx + 3 < tiles) _WinoTopTm[base + 3] = sum3;
}

void AexisWinograd23TransformOutputPack4_Impl(uint3 id)
{
    int gx = (int)id.x;
    int gy = (int)id.y;
    int gz = (int)id.z;
    if (gx >= _WinoBlockX || gy >= _WinoBlockY || gz >= _OutPacks)
        return;

    int vTmOffset = gz * _WinoOutCstep + gy * _WinoBlockX + gx;
    int packC = _OutPacks;

    float4 v00 = _WinoTopTm[vTmOffset + 0 * _WinoOutCstep * packC];
    float4 v01 = _WinoTopTm[vTmOffset + 1 * _WinoOutCstep * packC];
    float4 v02 = _WinoTopTm[vTmOffset + 2 * _WinoOutCstep * packC];
    float4 v03 = _WinoTopTm[vTmOffset + 3 * _WinoOutCstep * packC];
    float4 v10 = _WinoTopTm[vTmOffset + 4 * _WinoOutCstep * packC];
    float4 v11 = _WinoTopTm[vTmOffset + 5 * _WinoOutCstep * packC];
    float4 v12 = _WinoTopTm[vTmOffset + 6 * _WinoOutCstep * packC];
    float4 v13 = _WinoTopTm[vTmOffset + 7 * _WinoOutCstep * packC];
    float4 v20 = _WinoTopTm[vTmOffset + 8 * _WinoOutCstep * packC];
    float4 v21 = _WinoTopTm[vTmOffset + 9 * _WinoOutCstep * packC];
    float4 v22 = _WinoTopTm[vTmOffset + 10 * _WinoOutCstep * packC];
    float4 v23 = _WinoTopTm[vTmOffset + 11 * _WinoOutCstep * packC];
    float4 v30 = _WinoTopTm[vTmOffset + 12 * _WinoOutCstep * packC];
    float4 v31 = _WinoTopTm[vTmOffset + 13 * _WinoOutCstep * packC];
    float4 v32 = _WinoTopTm[vTmOffset + 14 * _WinoOutCstep * packC];
    float4 v33 = _WinoTopTm[vTmOffset + 15 * _WinoOutCstep * packC];

    float4 m00 = v00 + v01 + v02;
    float4 m01 = v10 + v11 + v12;
    float4 m02 = v20 + v21 + v22;
    float4 m03 = v30 + v31 + v32;

    float4 m10 = v01 - v02 + v03;
    float4 m11 = v11 - v12 + v13;
    float4 m12 = v21 - v22 + v23;
    float4 m13 = v31 - v32 + v33;

    if (_WinoBiasTerm != 0)
    {
        float4 biasValue = _WinoBias4[gz];
        v00 = biasValue + m00 + m01 + m02;
        v10 = biasValue + m10 + m11 + m12;
        v01 = biasValue + m01 - m02 + m03;
        v11 = biasValue + m11 - m12 + m13;
    }
    else
    {
        v00 = m00 + m01 + m02;
        v10 = m10 + m11 + m12;
        v01 = m01 - m02 + m03;
        v11 = m11 - m12 + m13;
    }

    v00 = AexisApplyActivation(v00);
    v01 = AexisApplyActivation(v01);
    v10 = AexisApplyActivation(v10);
    v11 = AexisApplyActivation(v11);
    v00 = AexisApplyFusedConvPRelu(v00, gz);
    v01 = AexisApplyFusedConvPRelu(v01, gz);
    v10 = AexisApplyFusedConvPRelu(v10, gz);
    v11 = AexisApplyFusedConvPRelu(v11, gz);

    int x = gx * 2;
    int y = gy * 2;
    _WinoOutArr[int3(x + 0, y + 0, gz)] = v00;
    if (x + 1 < _WinoOutW) _WinoOutArr[int3(x + 1, y + 0, gz)] = v01;
    if (y + 1 < _WinoOutH) _WinoOutArr[int3(x + 0, y + 1, gz)] = v10;
    if (y + 1 < _WinoOutH && x + 1 < _WinoOutW) _WinoOutArr[int3(x + 1, y + 1, gz)] = v11;
}

void AexisConv1x1Pack4_Impl(uint3 id)
{
    uint ow, oh, od;
    _ConvOutArr.GetDimensions(ow, oh, od);

    int ox = (int)id.x * 2;
    int oy = (int)id.y * 2;
    int op = (int)id.z * 2;
    if (ox >= (int)ow || oy >= (int)oh || op < 0 || op >= _OutPacks)
        return;

    int op1 = op + 1;
    bool hasX1 = ox + 1 < (int)ow;
    bool hasY1 = oy + 1 < (int)oh;
    bool hasOp1 = op1 < _OutPacks;

    float4 bias0 = _ConvB4[op];
    float4 bias1 = hasOp1 ? _ConvB4[op1] : 0.0;

    float4 sum00 = bias0;
    float4 sum01 = bias0;
    float4 sum10 = bias0;
    float4 sum11 = bias0;

    float4 sum20 = bias1;
    float4 sum21 = bias1;
    float4 sum30 = bias1;
    float4 sum31 = bias1;

    for (int ip0 = 0; ip0 < _InPacks; ip0++)
    {
        float4 v00 = _ConvInArr[int3(ox, oy, ip0)];
        float4 v01 = hasX1 ? _ConvInArr[int3(ox + 1, oy, ip0)] : 0.0;
        float4 v10 = hasY1 ? _ConvInArr[int3(ox, oy + 1, ip0)] : 0.0;
        float4 v11 = (hasX1 && hasY1) ? _ConvInArr[int3(ox + 1, oy + 1, ip0)] : 0.0;

        int base0 = ((op * _InPacks + ip0) * 4);
        float4 w00 = AexisReadPack4ConvWeight4(base0 + 0);
        float4 w01 = AexisReadPack4ConvWeight4(base0 + 1);
        float4 w02 = AexisReadPack4ConvWeight4(base0 + 2);
        float4 w03 = AexisReadPack4ConvWeight4(base0 + 3);

        sum00.x += dot(v00, w00);
        sum00.y += dot(v00, w01);
        sum00.z += dot(v00, w02);
        sum00.w += dot(v00, w03);

        if (hasX1)
        {
            sum01.x += dot(v01, w00);
            sum01.y += dot(v01, w01);
            sum01.z += dot(v01, w02);
            sum01.w += dot(v01, w03);
        }

        if (hasY1)
        {
            sum10.x += dot(v10, w00);
            sum10.y += dot(v10, w01);
            sum10.z += dot(v10, w02);
            sum10.w += dot(v10, w03);
        }

        if (hasX1 && hasY1)
        {
            sum11.x += dot(v11, w00);
            sum11.y += dot(v11, w01);
            sum11.z += dot(v11, w02);
            sum11.w += dot(v11, w03);
        }

        if (!hasOp1)
            continue;

        int base1 = ((op1 * _InPacks + ip0) * 4);
        float4 w10 = AexisReadPack4ConvWeight4(base1 + 0);
        float4 w11 = AexisReadPack4ConvWeight4(base1 + 1);
        float4 w12 = AexisReadPack4ConvWeight4(base1 + 2);
        float4 w13 = AexisReadPack4ConvWeight4(base1 + 3);

        sum20.x += dot(v00, w10);
        sum20.y += dot(v00, w11);
        sum20.z += dot(v00, w12);
        sum20.w += dot(v00, w13);

        if (hasX1)
        {
            sum21.x += dot(v01, w10);
            sum21.y += dot(v01, w11);
            sum21.z += dot(v01, w12);
            sum21.w += dot(v01, w13);
        }

        if (hasY1)
        {
            sum30.x += dot(v10, w10);
            sum30.y += dot(v10, w11);
            sum30.z += dot(v10, w12);
            sum30.w += dot(v10, w13);
        }

        if (hasX1 && hasY1)
        {
            sum31.x += dot(v11, w10);
            sum31.y += dot(v11, w11);
            sum31.z += dot(v11, w12);
            sum31.w += dot(v11, w13);
        }
    }

    sum00 = AexisApplyActivation(sum00);
    if (hasX1) sum01 = AexisApplyActivation(sum01);
    if (hasY1) sum10 = AexisApplyActivation(sum10);
    if (hasX1 && hasY1) sum11 = AexisApplyActivation(sum11);

    _ConvOutArr[int3(ox, oy, op)] = sum00;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op)] = sum01;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op)] = sum10;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op)] = sum11;

    if (!hasOp1)
        return;

    sum20 = AexisApplyActivation(sum20);
    if (hasX1) sum21 = AexisApplyActivation(sum21);
    if (hasY1) sum30 = AexisApplyActivation(sum30);
    if (hasX1 && hasY1) sum31 = AexisApplyActivation(sum31);

    _ConvOutArr[int3(ox, oy, op1)] = sum20;
    if (hasX1) _ConvOutArr[int3(ox + 1, oy, op1)] = sum21;
    if (hasY1) _ConvOutArr[int3(ox, oy + 1, op1)] = sum30;
    if (hasX1 && hasY1) _ConvOutArr[int3(ox + 1, oy + 1, op1)] = sum31;
}

void AexisPackRgbToPack4Gfpgan_Impl(uint3 id)
{
    uint w, h, d;
    _AexisOutArr.GetDimensions(w, h, d);
    if (id.x >= w || id.y >= h) return;
    uint tw, th;
    _AexisIn.GetDimensions(tw, th);
    int sx = (int) (id.x + _OffsetX) * _ScaleX;
    int sy = (int) (id.y + _OffsetY) * _ScaleY;
    sx = clamp(sx, 0, (int)tw - 1);
    sy = clamp(sy, 0, (int)th - 1);
    if (_FlipY != 0)
        sy = (int)th - 1 - sy;
    float4 v = _AexisIn[int2(sx, sy)];
    v.xyz = v.xyz * _InputValueScale + _InputValueBias;
    v.xyz = v.xyz * 2.0 - 1.0;
    _AexisOutArr[int3((int)id.x, (int)id.y, 0)] = float4(v.x, v.y, v.z, 0.0);
}

// P2 one-to-one depthwise CDHW convolution. The source and result are Pack4
// Texture2DArray slices [z * packs + pack]; only immutable vectorized weights
// use a StructuredBuffer. There is no activation buffer materialization.
void AexisConvDepthWise3dPack4CDHW_Impl(uint3 id)
{
    uint outWidth, outHeight, outSlices;
    _ConvOutArr.GetDimensions(outWidth, outHeight, outSlices);
    if (id.x >= outWidth || id.y >= outHeight || id.z >= outSlices)
        return;

    int packCount = max(1, _OutPacks);
    int outZ = (int)id.z / packCount;
    int outputPack = (int)id.z - outZ * packCount;
    if (outZ < 0 || outZ >= _OutD || outputPack < 0 || outputPack >= packCount)
        return;

    float4 sum = _DwConvB4[outputPack];
    int kernelPlane = _KernelWVar * _KernelHVar;
    int kernelVolume = kernelPlane * _KernelDVar;
    for (int kz = 0; kz < _KernelDVar; kz++)
    {
        int inputZ = outZ * _StrideDVar - _PadFrontVar + kz * _DilationDVar;
        if (inputZ < 0 || inputZ >= _InD)
            continue;
        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int inputY = (int)id.y * _StrideHVar - _PadTopVar + ky * _DilationHVar;
            if (inputY < 0 || inputY >= _InH)
                continue;
            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int inputX = (int)id.x * _StrideWVar - _PadLeftVar + kx * _DilationWVar;
                if (inputX < 0 || inputX >= _InW)
                    continue;
                int kernelIndex = (kz * kernelPlane) + (ky * _KernelWVar) + kx;
                float4 weights = _DwConvW4[outputPack * kernelVolume + kernelIndex];
                sum += _ConvInArr[int3(inputX, inputY, inputZ * packCount + outputPack)] * weights;
            }
        }
    }

    sum = AexisApplyActivation(sum);
    [unroll]
    for (int lane = 0; lane < 4; lane++)
        if (outputPack * 4 + lane >= _OutC)
            AexisWriteLane(sum, lane, 0.0);
    _ConvOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = sum;
}

// P2 one-to-one depthwise CDHW transposed convolution. Activations remain
// Pack4 Texture2DArray slices [z * packs + pack]; only immutable vectorized
// weights/bias are structured buffers.
void AexisDeconvolutionDepthWise3dPack4CDHW_Impl(uint3 id)
{
    uint outWidth, outHeight, outSlices;
    _ConvOutArr.GetDimensions(outWidth, outHeight, outSlices);
    if (id.x >= outWidth || id.y >= outHeight || id.z >= outSlices)
        return;

    int packCount = max(1, _OutPacks);
    int outZ = (int)id.z / packCount;
    int outputPack = (int)id.z - outZ * packCount;
    if (outZ < 0 || outZ >= _OutD || outputPack < 0 || outputPack >= packCount)
        return;

    float4 sum = _DwConvB4[outputPack];
    int kernelPlane = _KernelWVar * _KernelHVar;
    int kernelVolume = kernelPlane * _KernelDVar;
    int borderedZ = outZ + _PadFrontVar;
    int borderedY = (int)id.y + _PadTopVar;
    int borderedX = (int)id.x + _PadLeftVar;
    for (int kz = 0; kz < _KernelDVar; kz++)
    {
        int inputZNumerator = borderedZ - kz * _DilationDVar;
        if (inputZNumerator < 0 || (inputZNumerator % _StrideDVar) != 0)
            continue;
        int inputZ = inputZNumerator / _StrideDVar;
        if (inputZ < 0 || inputZ >= _InD)
            continue;
        for (int ky = 0; ky < _KernelHVar; ky++)
        {
            int inputYNumerator = borderedY - ky * _DilationHVar;
            if (inputYNumerator < 0 || (inputYNumerator % _StrideHVar) != 0)
                continue;
            int inputY = inputYNumerator / _StrideHVar;
            if (inputY < 0 || inputY >= _InH)
                continue;
            for (int kx = 0; kx < _KernelWVar; kx++)
            {
                int inputXNumerator = borderedX - kx * _DilationWVar;
                if (inputXNumerator < 0 || (inputXNumerator % _StrideWVar) != 0)
                    continue;
                int inputX = inputXNumerator / _StrideWVar;
                if (inputX < 0 || inputX >= _InW)
                    continue;

                int kernelIndex = (kz * kernelPlane) + (ky * _KernelWVar) + kx;
                float4 weights = _DwConvW4[outputPack * kernelVolume + kernelIndex];
                sum += _ConvInArr[int3(inputX, inputY, inputZ * packCount + outputPack)] * weights;
            }
        }
    }

    sum = AexisApplyActivation(sum);
    [unroll]
    for (int lane = 0; lane < 4; lane++)
        if (outputPack * 4 + lane >= _OutC)
            AexisWriteLane(sum, lane, 0.0);
    _ConvOutArr[int3((int)id.x, (int)id.y, (int)id.z)] = sum;
}

void AexisGfpganStyleModulation_Impl(uint3 id)
{
    int outputChannel = (int)id.x;
    if (outputChannel < 0 || outputChannel >= _GfpganStyleOutputDim)
        return;

    float sum = _GfpganModulationB[outputChannel];
    int weightOffset = outputChannel * _GfpganStyleInputDim;
    for (int inputChannel = 0; inputChannel < _GfpganStyleInputDim; inputChannel++)
        sum += _GfpganModulationW[weightOffset + inputChannel] * _GfpganStyleIn[int2(inputChannel, _GfpganStyleRow)];
    _GfpganStyleOut[int2(outputChannel, 0)] = sum;
}

void AexisGfpganStyleDemod_Impl(uint3 id)
{
    int outputChannel = (int)id.x;
    if (outputChannel < 0 || outputChannel >= _GfpganOutputChannels)
        return;

    float sum = 0.0;
    int outputBase = outputChannel * _GfpganHiddenDim * _GfpganKernelArea;
    for (int inputChannel = 0; inputChannel < _GfpganHiddenDim; inputChannel++)
    {
        float style = _GfpganStyleVector[int2(inputChannel, 0)];
        int inputBase = outputBase + inputChannel * _GfpganKernelArea;
        for (int kernel = 0; kernel < _GfpganKernelArea; kernel++)
        {
            float value = _GfpganSelfWeight[inputBase + kernel] * style;
            sum += value * value;
        }
    }
    _GfpganDemodOut[int2(outputChannel, 0)] = rsqrt(sum + 1e-8);
}

void AexisGfpganBuildDynamicWeight_Impl(uint3 id)
{
    int index = (int)id.x;
    int outputPacks = (_GfpganOutputChannels + 3) / 4;
    int inputPacks = _GfpganHiddenDim / 4;
    int total = outputPacks * inputPacks * _GfpganKernelArea * 4;
    if (index < 0 || index >= total)
        return;

    int outputLane = index % 4;
    int q = index / 4;
    int kernel = q % _GfpganKernelArea;
    q /= _GfpganKernelArea;
    int inputPack = q % inputPacks;
    int outputPack = q / inputPacks;
    int outputChannel = outputPack * 4 + outputLane;

    float4 weight = 0.0;
    if (outputChannel < _GfpganOutputChannels)
    {
        float demod = _GfpganUseDemod != 0 ? _GfpganDemodIn[int2(outputChannel, 0)] : 1.0;
        int inputBase = inputPack * 4;
        int outputBase = outputChannel * _GfpganHiddenDim * _GfpganKernelArea;
        weight.x = _GfpganSelfWeight[outputBase + (inputBase + 0) * _GfpganKernelArea + kernel] * _GfpganStyleVector[int2(inputBase + 0, 0)] * demod;
        weight.y = _GfpganSelfWeight[outputBase + (inputBase + 1) * _GfpganKernelArea + kernel] * _GfpganStyleVector[int2(inputBase + 1, 0)] * demod;
        weight.z = _GfpganSelfWeight[outputBase + (inputBase + 2) * _GfpganKernelArea + kernel] * _GfpganStyleVector[int2(inputBase + 2, 0)] * demod;
        weight.w = _GfpganSelfWeight[outputBase + (inputBase + 3) * _GfpganKernelArea + kernel] * _GfpganStyleVector[int2(inputBase + 3, 0)] * demod;
    }

    _GfpganDynW4Tex[int3(index % _GfpganDynW4Width, index / _GfpganDynW4Width, 0)] = weight;
}

uint AexisGfpganNoiseHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

// Channel PReLU fused into the direct Pack4 3x3 convolution. The immutable
// slope buffer is shared with the standalone PReLU kernel; keeping the math in
// this final write step avoids a separate full-resolution feature-map pass.
float4 AexisApplyFusedConvPRelu(float4 value, int outputPack)
{
    if (_ConvUsePRelu == 0)
        return value;

    int baseChannel = outputPack * 4;
    int slopeCount = max(1, _PReluSlopePack4Count);
    float4 slope = slopeCount == 1
        ? _PReluSlopePack4Buf[0].xxxx
        : float4(
            _PReluSlopePack4Buf[clamp(baseChannel + 0, 0, slopeCount - 1)],
            _PReluSlopePack4Buf[clamp(baseChannel + 1, 0, slopeCount - 1)],
            _PReluSlopePack4Buf[clamp(baseChannel + 2, 0, slopeCount - 1)],
            _PReluSlopePack4Buf[clamp(baseChannel + 3, 0, slopeCount - 1)]);
    return max(value, 0.0) + min(value, 0.0) * slope;
}

void AexisGfpganAddNoisePack4_Impl(uint3 id)
{
    uint width, height, depth;
    _GfpganNoiseInOutArr.GetDimensions(width, height, depth);
    if (id.x >= width || id.y >= height || id.z >= depth)
        return;

    uint pixel = id.y * width + id.x;
    uint seed = (uint)_GfpganNoiseSeed;
    float u1 = max(1e-7, ((float)AexisGfpganNoiseHash(pixel ^ seed) + 1.0) * (1.0 / 4294967296.0));
    float u2 = ((float)AexisGfpganNoiseHash(pixel + 0x9e3779b9u + seed) + 1.0) * (1.0 / 4294967296.0);
    float noise = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2) * _GfpganNoiseWeight;
    _GfpganNoiseInOutArr[int3((int)id.x, (int)id.y, (int)id.z)] += noise;
}
