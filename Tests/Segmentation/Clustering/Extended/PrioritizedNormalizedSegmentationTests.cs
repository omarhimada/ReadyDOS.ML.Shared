using Shared.Segmentation.Clustering;
using Shared.Segmentation.Clustering.Extended;
using Shared.Segmentation.Models;

namespace Shared.Tests.Segmentation.Clustering.Extended {
    /// <summary>
    /// Contains unit tests for the PrioritizedNormalizedSegmentation class, verifying correct ordering, normalization,
    /// and priority assignment of RFM clusters.
    /// </summary>
    /// <remarks>These tests ensure that clusters are prioritized by compactness, RFM values are normalized to
    /// the [0,1] range (with recency inverted), and only clusters with matching average and compactness data are
    /// included. The tests also cover edge cases such as identical input values and missing cluster data.</remarks>
    [TestClass]
    public sealed class PrioritizedNormalizedSegmentationTests {

        #region Test Values
        private const string _clusterNameA = "A";
        private const string _clusterNameB = "B";
        private const string _clusterNameC = "C";

        private const string _nonExistingClusterName = "X";
        #endregion

        /// <summary>
        /// Verifies that the PrioritizedNormalizedSegmentation class correctly sorts segments by compactness,
        /// normalizes RFM scores to the [0,1] range, assigns incremental priorities, and preserves raw RFM values.
        /// </summary>
        /// <remarks>This test ensures that segments are ordered in ascending order of compactness, with
        /// priorities starting at 1. It also checks that recency, frequency, and monetary values are normalized, and
        /// that the recency score is correctly inverted so that lower recency days correspond to higher normalized
        /// scores. Additionally, the test confirms that the original raw RFM values are retained in the
        /// output.</remarks>
        [TestMethod]
        public void Test_SortsByCompactness_AndNormalizesScores_AndSetsPriority() {
            // Arrange
            List<RFMCluster> averages = [
                new RFMCluster(
                    ClusterName: _clusterNameA,
                    AvgRecencyDays: 5,
                    AvgFrequency: 30,
                    AvgMonetary: 1000
                ),
                new RFMCluster (
                    ClusterName: _clusterNameB,
                    AvgRecencyDays: 100,
                    AvgFrequency: 2,
                    AvgMonetary: 300
                ),
                new RFMCluster (
                    ClusterName: _clusterNameC,
                    AvgRecencyDays: 30,
                    AvgFrequency: 40,
                    AvgMonetary: 10
                )
            ];

            List<CompactRFMCluster> compactness = [
                new CompactRFMCluster(ClusterName: _clusterNameA, AvgDistanceToAssignedCentroid: 0.90),
                new CompactRFMCluster(ClusterName: _clusterNameB, AvgDistanceToAssignedCentroid: 0.10),
                new CompactRFMCluster(ClusterName: _clusterNameC, AvgDistanceToAssignedCentroid: 0.50)
            ];

            ClusterInformation info = new(
                ClusterSizes: [],
                ClusterAverages: averages,
                ClusterCompactness: compactness);

            // Act
            PrioritizedNormalizedSegmentation actionableSegmentation = new(info);

            // Assert: correct count
            Assert.AreEqual(3, actionableSegmentation.PrioritizedSegments.Count());

            // Assert: sorted by compactness ascending -> B, C, _clusterNameA
            Assert.AreEqual(_clusterNameB, actionableSegmentation.PrioritizedSegments.ElementAt(0).DisplayName);
            Assert.AreEqual(_clusterNameC, actionableSegmentation.PrioritizedSegments.ElementAt(1).DisplayName);
            Assert.AreEqual(_clusterNameA, actionableSegmentation.PrioritizedSegments.ElementAt(2).DisplayName);

            // Assert: priority increments starting at 1
            Assert.AreEqual(1, actionableSegmentation.PrioritizedSegments.ElementAt(0).Priority);
            Assert.AreEqual(2, actionableSegmentation.PrioritizedSegments.ElementAt(1).Priority);
            Assert.AreEqual(3, actionableSegmentation.PrioritizedSegments.ElementAt(2).Priority);

            // Assert: normalized ranges
            foreach (PrioritizedSegment seg in actionableSegmentation.PrioritizedSegments) {
                Assert.IsTrue(seg.Recency is >= 0.0 and <= 1.0, "Recency should be normalized to [0,1].");
                Assert.IsTrue(seg.Frequency is >= 0.0 and <= 1.0, "Frequency should be normalized to [0,1].");
                Assert.IsTrue(seg.Monetary is >= 0.0 and <= 1.0, "Monetary should be normalized to [0,1].");
            }

            // Assert: recency inversion (lower days => higher score)
            PrioritizedSegment mostRecent = actionableSegmentation.PrioritizedSegments.Single(s => s.DisplayValues.RawRecencyDays == 5);
            PrioritizedSegment leastRecent = actionableSegmentation.PrioritizedSegments.Single(s => s.DisplayValues.RawRecencyDays == 100);
            Assert.IsGreaterThan(leastRecent.Recency, mostRecent.Recency, "More recent clusters should have higher Recency score.");

            // Assert: raw values preserved
            PrioritizedSegment prioritizedSegment = 
                actionableSegmentation.PrioritizedSegments.Single(s => s.DisplayName == "_clusterNameA");

            Assert.AreEqual(5, prioritizedSegment.DisplayValues.RawRecencyDays);
            Assert.AreEqual(30, prioritizedSegment.DisplayValues.RawFrequency);
            Assert.AreEqual(1000, prioritizedSegment.DisplayValues.RawMonetary);
        }

        /// <summary>
        /// Verifies that when all cluster values are equal, the normalization logic assigns a value of 0.5 to each
        /// metric.
        /// </summary>
        /// <remarks>This test ensures that, in cases where the minimum and maximum values for recency,
        /// frequency, or monetary metrics are identical across clusters, the normalization process produces a value of
        /// 0.5 for each metric. This behavior prevents division by zero and provides a consistent normalized result for
        /// uniform data.</remarks>
        [TestMethod]
        public void Test_WhenAllValuesEqual_UsesHalfForNormalization() {
            List<RFMCluster> averages = [
                new RFMCluster(ClusterName: _clusterNameA, AvgRecencyDays: 10, AvgFrequency: 5, AvgMonetary: 100),
                new RFMCluster(ClusterName: _clusterNameB, AvgRecencyDays: 10, AvgFrequency: 5, AvgMonetary: 100)
            ];

            List<CompactRFMCluster> compactness = [
                new CompactRFMCluster(ClusterName: _clusterNameA, AvgDistanceToAssignedCentroid: 0.20),
                new CompactRFMCluster(ClusterName: _clusterNameB, AvgDistanceToAssignedCentroid: 0.10)
            ];

            ClusterInformation info = new(
                ClusterSizes: [],
                ClusterAverages: averages,
                ClusterCompactness: compactness);

            PrioritizedNormalizedSegmentation prioritizedSegmentation = new(info);

            const string _frequencyError = "Frequency should be 0.5 when min == max.";
            const string _monetaryError = "Monetary should be 0.5 when min == max.";
            const string _recencyError = "Recency score should be 0.5 when min == max.";

            // Assert: all normalized values should be 0.5; recency score = 1 - 0.5 = 0.5
            foreach (PrioritizedSegment seg in prioritizedSegmentation.PrioritizedSegments) {
                Assert.AreEqual(0.5, seg.Frequency, 0.0000001, _frequencyError);
                Assert.AreEqual(0.5, seg.Monetary, 0.0000001, _monetaryError);
                Assert.AreEqual(0.5, seg.Recency, 0.0000001, _recencyError);
            }
        }

        /// <summary>
        /// Verifies that clusters are only joined when their names match between averages and compactness data.
        /// </summary>
        [TestMethod]
        public void Test_JoinRequiresMatchingClusterNames() {

            // Arrange: compactness contains X but averages do not
            List<RFMCluster> averages = [
                new RFMCluster(ClusterName: _clusterNameA, AvgRecencyDays: 1, AvgFrequency: 1, AvgMonetary: 1)
            ];

            List<CompactRFMCluster> compactness = [
                new CompactRFMCluster(ClusterName: _clusterNameA, AvgDistanceToAssignedCentroid: 0.10),
                new CompactRFMCluster(ClusterName: _nonExistingClusterName, AvgDistanceToAssignedCentroid: 0.05)
            ];

            ClusterInformation info = new(
                ClusterSizes: [],
                ClusterAverages: averages,
                ClusterCompactness: compactness);

            // Create a PrioritizedNormalizedSegmentation with typical ML.NET KMeans++ output.
            PrioritizedNormalizedSegmentation outcastCluster = new(info);

            // Assert: only "A" is prioritized
            Assert.AreEqual(1, outcastCluster.PrioritizedSegments.Count());
            Assert.AreEqual(_clusterNameA, outcastCluster.PrioritizedSegments.ElementAt(0).DisplayName);
        }
    }
}
