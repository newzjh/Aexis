#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.Linq;
using NUnit.Framework;

public sealed class AIImageModelReleaseMappingTests
{
    [Test]
    public void MainModelGroupsMapEveryRequiredFileToThePublishedRelease()
    {
        AssertReleaseGroup(AIImageModelGroupId.ClipMobileClipS0, "clip",
            "bpe_simple_vocab_16e6.txt",
            "vocab.txt",
            "mobileclip_s0_export.label_embeddings.json",
            "image_encoder.ncnn.param",
            "image_encoder.ncnn.bin");
        AssertReleaseGroup(AIImageModelGroupId.CodeFormerDefault, "codeformer",
            "encoder.param",
            "encoder.bin",
            "generator.param",
            "generator.bin",
            "yolov7-lite-e.param",
            "yolov7-lite-e.bin");
        AssertReleaseGroup(AIImageModelGroupId.Matting, "matting", "matting.param", "matting.bin");
        AssertReleaseGroup(AIImageModelGroupId.YoloV8PersonSegmentation, "yolo",
            "yolov8n_seg.ncnn.param",
            "yolov8n_seg.ncnn.bin");
    }

    [Test]
    public void StableDiffusionFilesUseTheirTwoPublishedReleaseTags()
    {
        var group = AIImageModelDelivery.GetGroup(AIImageModelGroupId.StableDiffusion);

        Assert.That(AIImageModelDelivery.CanDownloadGroup(group), Is.True);
        Assert.That(group.RemoteFiles.Count, Is.EqualTo(group.Files.Count));
        Assert.That(group.RemoteFiles.Select(file => file.ReleaseTag).Distinct().OrderBy(tag => tag),
            Is.EqualTo(new[] { "model", "sdinpainting" }));
    }

    [Test]
    public void GroupsWithoutCompletePublishedFilesAreUnavailable()
    {
        Assert.That(AIImageModelDelivery.CanDownloadGroup(
            AIImageModelDelivery.GetGroup(AIImageModelGroupId.Yolo11PersonSegmentation)), Is.False);
        Assert.That(AIImageModelDelivery.CanDownloadGroup(
            AIImageModelDelivery.GetGroup(AIImageModelGroupId.Qwen35FullPrecision)), Is.False);
    }

    [Test]
    public void ReleaseUrlsUseTheCurrentAexisRepositoryAndExactAssetName()
    {
        const string baseUrlVariable = "AIIMAGE_MODEL_RELEASE_BASE_URL";
        var previousBaseUrl = Environment.GetEnvironmentVariable(baseUrlVariable);
        try
        {
            Environment.SetEnvironmentVariable(baseUrlVariable, null);
            Assert.That(AIImageModelDelivery.GetReleaseAssetUrl("matting.bin", "matting"),
                Is.EqualTo("https://github.com/newzjh/Aexis/releases/download/matting/matting.bin"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(baseUrlVariable, previousBaseUrl);
        }
    }

    private static void AssertReleaseGroup(AIImageModelGroupId id, string releaseTag, params string[] assetNames)
    {
        var group = AIImageModelDelivery.GetGroup(id);

        Assert.That(AIImageModelDelivery.CanDownloadGroup(group), Is.True);
        Assert.That(group.RemoteFiles.Count, Is.EqualTo(group.Files.Count));
        Assert.That(group.RemoteFiles.Select(file => file.ReleaseTag).Distinct(), Is.EqualTo(new[] { releaseTag }));
        Assert.That(group.RemoteFiles.Select(file => file.AssetName), Is.EquivalentTo(assetNames));
        Assert.That(group.RemoteFiles.Select(file => file.DestinationPath), Is.EquivalentTo(group.Files));
    }
}
#endif
