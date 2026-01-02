namespace Shared.Segmentation.Models {
    /// <summary>
    /// Represents the average recency, frequency, and monetary values for a customer cluster.
    /// Used during KMeans++ segmentation analysis.
    /// </summary>
    /// <param name="ClusterName">The name of the customer cluster for which the averages are calculated.</param>
    /// <param name="AvgRecencyDays">The average number of days since the last activity for members of the cluster.</param>
    /// <param name="AvgFrequency">The average frequency of activity for members of the cluster.</param>
    /// <param name="AvgMonetary">The average monetary value associated with members of the cluster.</param>
    public sealed record RFMCluster(
        string ClusterName,
        double AvgRecencyDays,
        double AvgFrequency,
        double AvgMonetary);
}
