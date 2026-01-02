using Microsoft.ML;
using Shared.Segmentation.Clustering;

namespace Shared.Segmentation.Models {
    /// <summary>
    /// Output from KMeans++ segmentation with ML.NET.
    /// </summary>
    /// <param name="Model"></param>
    /// <param name="ClusterSizes"></param>
    /// <param name="ClusterAverages"></param>
    /// <param name="ClusterCompactness"></param>
    public sealed record SegmentationResult(
        ITransformer Model,
        IReadOnlyList<Cluster> ClusterSizes,
        IReadOnlyList<RFMCluster> ClusterAverages,
        IReadOnlyList<CompactRFMCluster> ClusterCompactness);
}
