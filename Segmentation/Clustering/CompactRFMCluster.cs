namespace ReadyDOS.Shared.Segmentation.Clustering {

    /// <summary>
    /// Represents the compactness metrics for a cluster, including its name and the average distance of its members to
    /// the assigned centroid.
    /// </summary>
    /// <remarks>This data transfer object is typically used to convey clustering quality metrics in analytics
    /// or reporting scenarios. _clusterNameA lower average distance indicates a more compact cluster.</remarks>
    /// <param name="ClusterName">The name of the cluster for which compactness metrics are reported. Cannot be null or empty.</param>
    /// <param name="AvgDistanceToAssignedCentroid">The average distance of all points in the cluster to their assigned centroid. Must be greater than or equal to
    /// zero.</param>
    public sealed record CompactRFMCluster(
        string ClusterName,
        double AvgDistanceToAssignedCentroid);

}
