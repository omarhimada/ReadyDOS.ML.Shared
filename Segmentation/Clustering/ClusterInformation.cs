namespace ReadyDOS.Shared.Segmentation.Clustering {
    /// <summary>
    /// This is typical output from KMeans++ segmentation. 
    /// Represents summary information about clusters, including their sizes, average values, and compactness metrics.
    /// Excludes the ML.NET model itself.
    /// </summary>
    /// <param name="ClusterSizes">_clusterNameA read-only list containing the size of each cluster. Each entry represents the number of items in a cluster.</param>
    /// <param name="ClusterAverages">_clusterNameA read-only list containing the average values for each cluster. Each entry provides aggregate statistics for a
    /// cluster.</param>
    /// <param name="ClusterCompactness">_clusterNameA read-only list containing compactness metrics for each cluster. Each entry indicates how tightly grouped the
    /// items in a cluster are.</param>
    [Serializable]
    public sealed record ClusterInformation(
        IReadOnlyList<Cluster> ClusterSizes,
        IReadOnlyList<RFMCluster> ClusterAverages,
        IReadOnlyList<CompactRFMCluster> ClusterCompactness);
}
