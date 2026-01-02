using Shared.Segmentation.Models;

namespace Shared.Segmentation.Clustering.Extended {
    /// <summary>
    /// Represents a segmentation of clusters with normalized and prioritized segments, enabling consistent customer
    /// segment analysis across different datasets.
    /// </summary>
    /// <remarks>This class normalizes recency, frequency, and monetary values for each segment, allowing for
    /// meaningful comparison and prioritization regardless of the underlying dataset scale (such as between discount
    /// and luxury retailers). Segment names and priorities are assigned based on normalized values to provide
    /// consistent interpretation across use cases.</remarks>
    public class PrioritizedNormalizedSegmentation {
        /// <summary>
        /// Gets or sets the collection of prioritized customer/user segments to be processed.
        /// </summary>
        public IEnumerable<PrioritizedSegment> PrioritizedSegments { get; set; } = [];

        /// <summary>
        /// Due to normalization you can assign consistent, meaningful names to segments across datasets.
        /// Some orgnizatios miight not want to refer to their customers as "VIP Elite" or "Sleeping Spenders", for example.
        /// </summary>
        public enum SegmentRepresentation {
            VIPElite,
            PremiumEngaged,
            SleepingSpenders,
            WinBackCandidates,
            EngagementRichValueLow,
            MassUpsellPool,
            Unnamed
        }

        /// <summary>
        /// Provides display names for customer segments. 
        /// Override this virtual property to customize segment labels for your organization.
        /// </summary>
         public virtual Dictionary<SegmentRepresentation, string> SegmentDisplayNameMap => new() {
            { SegmentRepresentation.VIPElite, "Top Clients" },
            { SegmentRepresentation.PremiumEngaged, "Premium Engaged" },
            { SegmentRepresentation.SleepingSpenders, "Dormant Value" },
            { SegmentRepresentation.WinBackCandidates, "Win-Back Candidates" },
            { SegmentRepresentation.EngagementRichValueLow, "Engagement-Rich, Value-Low" },
            { SegmentRepresentation.MassUpsellPool, "Automated Upsell Targets" },
            { SegmentRepresentation.Unnamed, $"Emerging Segment" },

        };

        /// <summary>
        /// /// <summary>
        /// Implications regarding a business or organization's priorities can be derived from distance to centroid. 
        /// - Low  (e.g., 0.34) Very tight, very similar customers, Safe to automate, reliable for targeted campaigns
        /// - High (e.g., 1.13) Loose, mixed behavior. Flag it for analysis, test carefully, or re-segment.
        /// We can use this extend the typical segmentation output with prioritization. Normalizing values makes this
        /// reusable across businesses with varying datasets.
        /// </summary>
        /// <param name="clustering">Output from the Segmentation Workflow.</param>
        public PrioritizedNormalizedSegmentation(ClusterInformation clustering) {
            clustering.Deconstruct(
                out var clusterSizes,
                out var clusterAverages,
                out var clusterCompactness);

            #region Normalize the quantities for different datasets (e.g.: the dollar store vs. a luxury retailer)
            double minRecency = clusterAverages.Min(a => a.AvgRecencyDays);
            double maxRecency = clusterAverages.Max(a => a.AvgRecencyDays);

            double minFrequency = clusterAverages.Min(a => a.AvgFrequency);
            double maxFrequency = clusterAverages.Max(a => a.AvgFrequency);

            double minMonetary = clusterAverages.Min(a => a.AvgMonetary);
            double maxMonetary = clusterAverages.Max(a => a.AvgMonetary);
            #endregion

            static double MinMax(double v, double min, double max) =>
                max - min <= 0 ? 0.5 : (v - min) / (max - min);

            PrioritizedSegments =
                clusterCompactness
                .Join(clusterAverages, c => c.ClusterName, a => a.ClusterName, (c, a) => new { c, a })

                .OrderBy(x => x.c.AvgDistanceToAssignedCentroid)
                .Select((x, i) => {
                    double recencyNorm = MinMax(x.a.AvgRecencyDays, minRecency, maxRecency);
                    double recencyScore = 1.0 - recencyNorm;
                    double frequencyNorm = MinMax(x.a.AvgFrequency, minFrequency, maxFrequency);
                    double monetaryNorm = MinMax(x.a.AvgMonetary, minMonetary, maxMonetary);

                    // Due to normalization the assigned meaningful names are consistent across datasets
                    string displayName =
                        (recencyScore, frequencyNorm, monetaryNorm) switch {
                            var (r, f, m) when r > 0.7 && m > 0.7 && f > 0.6 => SegmentDisplayNameMap[SegmentRepresentation.VIPElite],
                            var (r, f, m) when r > 0.5 && m > 0.5 => SegmentDisplayNameMap[SegmentRepresentation.PremiumEngaged],
                            var (r, f, m) when r < 0.25 && m > 0.5 => SegmentDisplayNameMap[SegmentRepresentation.SleepingSpenders],
                            var (r, f, m) when r < 0.35 && m > 0.3 => SegmentDisplayNameMap[SegmentRepresentation.WinBackCandidates],
                            var (r, f, m) when f > 0.7 && m < 0.2 => SegmentDisplayNameMap[SegmentRepresentation.EngagementRichValueLow],
                            var (r, f, m) when m < 0.15 => SegmentDisplayNameMap[SegmentRepresentation.MassUpsellPool],

                            _ => $"{SegmentDisplayNameMap[SegmentRepresentation.Unnamed]} {i}"
                        };

                    return new PrioritizedSegment {
                        DisplayName = displayName,
                        Recency = recencyScore,
                        Frequency = frequencyNorm,
                        Monetary = monetaryNorm,
                        Priority = i + 1,

                        // Preserve raw values for display in UI
                        DisplayValues = new PrioritizedSegment.ForDisplayPurposes {
                            RawRecencyDays = x.a.AvgRecencyDays,
                            RawFrequency = x.a.AvgFrequency,
                            RawMonetary = x.a.AvgMonetary

                        }
                    };
                })
                .ToList();

        }
    }
}
