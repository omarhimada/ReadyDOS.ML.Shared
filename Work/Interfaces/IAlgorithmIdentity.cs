using System;
using System.Collections.Generic;
using System.Text;

namespace Shared.Work.Interfaces {
    /// <summary>
    /// Provides a set of algorithm identifiers and their associated display names supported by the system.
    /// </summary>
    /// <remarks>The set of supported algorithms is fixed at initialization and does not change at runtime.
    /// Use the SupportedAlgorithms dictionary to retrieve the human-readable name for a given algorithm
    /// identifier.</remarks>
    public interface IAlgorithmIdentity {

        /// <summary>
        /// Specifies the set of supported machine learning algorithms.
        /// </summary>
        /// <remarks>Use this enumeration to select the algorithm to be applied in machine learning
        /// workflows. The available values represent common algorithm types, such as matrix factorization, LBFGS
        /// optimization, k-means clustering, and forecasting. Not all algorithms may be implemented or available in
        /// every context; refer to the specific API documentation for details on supported operations.</remarks>
        public enum Algorithm {
            MatrixFactorization,
            LightGbmForecasting,
            LBFGSOptimization,
            KMeansClustering,
        }

        /// <summary>
        /// Specifies the types of jobs that can be executed in the system.
        /// </summary>
        public enum JobType {
            Recommendations,
            ChurnPrediction,
            Forecasting,
            Segmentation,
        }

        /// <summary>
        /// Provides a read-only mapping from job types to their corresponding machine learning algorithms.
        /// </summary>
        /// <remarks>This dictionary enables consumers to determine which algorithm is associated with a
        /// given job type. The mapping is static and does not change at runtime.</remarks>
        public static readonly IReadOnlyDictionary<JobType, Algorithm> JobTypeToAlgorithm = new Dictionary<JobType, Algorithm>
        {
            { JobType.Recommendations, Algorithm.MatrixFactorization },
            { JobType.ChurnPrediction, Algorithm.LBFGSOptimization },
            { JobType.Segmentation, Algorithm.KMeansClustering },
            { JobType.Forecasting, Algorithm.LightGbmForecasting }
        };

        /// <summary>
        /// Provides a mapping of supported algorithms to their display names.
        /// </summary>
        /// <remarks>The dictionary contains the set of algorithms recognized by the system and their
        /// corresponding human-readable names. The set of supported algorithms is fixed at initialization and does not
        /// change at runtime.</remarks>
        public static readonly Dictionary<Algorithm, string> SupportedAlgorithms = new() {
            { Algorithm.MatrixFactorization, @"Matrix Factorization" },
            { Algorithm.LBFGSOptimization, @"L-BFGS Optimization" },
            { Algorithm.KMeansClustering, @"K-Means++" },
            { Algorithm.LightGbmForecasting, @"LightGBM" }
        };

        /// <summary>
        /// Provides descriptive text for each supported job type, including business value and use case information.
        /// </summary>
        /// <remarks>Each entry maps a <see cref="JobType"/> value to a string that describes the purpose
        /// and business impact of the corresponding job. These descriptions can be used in user interfaces,
        /// documentation, or logging to help users understand the intent and benefits of each job type.</remarks>
        public static readonly Dictionary<JobType, string> JobTypeDescriptions = new()
        {
            { JobType.Recommendations, """
                Trains a recommendation model that suggests the most relevant products for each customer.
                Business value: increases Average Order Value (AOV), basket expansion, and purchase frequency,
                driving higher total revenue per user and improving repeat sales.
                """ },

            { JobType.ChurnPrediction, """
                Trains a binary classifier to detect customers likely to churn.
                Business value: reduces customer loss, protects recurring revenue, and maximizes
                Customer Lifetime Value (CLV) by optimizing retention and automating intervention.
                """ },

            { JobType.Segmentation, """
                Performs behavioral clustering to generate customer segments using K-Means++.
                Business value: enables targeted pricing, promotions, and smarter budget allocation,
                improving campaign conversion, discount efficiency, and incremental revenue lift.
                """ },

            { JobType.Forecasting, """
                Trains a high-accuracy forecasting model on tabular or time-series business data.
                Business value: informs price optimization, promotion timing, and inventory planning,
                maximizing revenue outcomes by modeling price × demand and preventing stockout losses.
                """ }
        };
    }
}
