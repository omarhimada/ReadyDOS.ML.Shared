using ReadyDOS.Shared.Segmentation.Clustering;

namespace ReadyDOS.Shared.Segmentation {
    /// <summary>
    /// Output from KMeans++ segmentation via ML.NET.
    /// </summary>
    /// <param name="Model"></param>
    /// <param name="ClusterSizes"></param>
    /// <param name="ClusterAverages"></param>
    /// <param name="ClusterCompactness"></param>
    public sealed record SegmentationResult(
        IReadOnlyList<Cluster> ClusterSizes,
        IReadOnlyList<RFMCluster> ClusterAverages,
        IReadOnlyList<CompactRFMCluster> ClusterCompactness);
}
