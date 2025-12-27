using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Shared.Work.Interfaces {
    public interface ISegmentation {
        /// <summary>
        /// Only exposes aggregated counts and distances, which are safe and useful for UI charts.
        /// </summary>
        public interface IKMeansSegmentationService<TInput, TPrediction>
            where TInput : class
            where TPrediction : class, new() {
            /// <summary>
            /// Trains a KMeans++ clustering model and returns cluster-labeled predictions.
            /// </summary>
            Task<IKMeansSegmentationResult<TInput, TPrediction>> TrainAndLabelAsync(
                MLContext mlContext,
                IEnumerable<TInput> data,
                int numberOfClusters,
                CancellationToken ct = default);

            /// <summary>
            /// Applies an already-trained clustering model to new data.
            /// </summary>
            Task<IKMeansSegmentationResult<TInput, TPrediction>> LabelAsync(
                ITransformer model,
                IEnumerable<TInput> newData,
                CancellationToken ct = default);
        }

        /// <summary>
        /// Container for cluster assignments and raw rows.
        /// </summary>
        public interface IKMeansSegmentationResult<TInput, TPrediction> {
            /// <summary>
            /// Gets the trained machine learning model used for making predictions.
            /// </summary>
            /// <remarks>The returned model implements the ITransformer interface and can be used to
            /// transform input data or make predictions. The model is typically produced by a training process and may
            /// be used for scoring or further evaluation.</remarks>
            ITransformer Model { get; }

            /// <summary>
            /// Gets the collection of input rows paired with their corresponding predictions.
            /// </summary>
            IReadOnlyList<(TInput Row, TPrediction Prediction)> LabeledRows { get; }

            /// <summary>
            /// Gets the collection of cluster size information for the current context.
            /// </summary>
            IReadOnlyList<ClusterSizeDto> ClusterSizes { get; }

            /// <summary>
            /// Gets the collection of centroid confidence values associated with the current entity.
            /// </summary>
            IReadOnlyList<CentroidConfidenceDto> CentroidConfidence { get; }
        }

        /// <summary>
        /// Represents the size of a cluster, including its name and the number of nodes it contains.
        /// </summary>
        /// <param name="ClusterName">The name of the cluster. Cannot be null or empty.</param>
        /// <param name="Count">The number of nodes in the cluster. Must be zero or greater.</param>
        public sealed record ClusterSizeDto(string clusterName, int count);

        /// <summary>
        /// Represents summary statistics describing the confidence of a centroid assignment within a cluster, including
        /// average, minimum, and maximum distances from the centroid.
        /// </summary>
        /// <remarks>These statistics can be used to assess the compactness and reliability of the cluster
        /// assignment. Lower average and maximum distances generally indicate higher confidence in the centroid's
        /// representation of the cluster.</remarks>
        /// <param name="ClusterId">The unique identifier of the cluster to which the centroid belongs.</param>
        /// <param name="AvgDistance">The average distance of all points in the cluster from the centroid.</param>
        /// <param name="MinDistance">The minimum distance from the centroid to any point in the cluster.</param>
        /// <param name="MaxDistance">The maximum distance from the centroid to any point in the cluster.</param>
        public sealed record CentroidConfidenceDto(uint clusterId, double avgDistance, double minDistance, double maxDistance);
    }
}
