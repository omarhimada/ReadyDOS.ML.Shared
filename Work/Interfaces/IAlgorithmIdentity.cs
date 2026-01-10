namespace ReadyDOS.Shared.Work.Interfaces {
    /// <summary>
    /// Provides a set of algorithm identifiers and their associated display names supported by the system.
    /// </summary>
    /// <remarks>The set of supported algorithms is fixed at initialization and does not change at runtime.
    /// Use the SupportedAlgorithms dictionary to retrieve the human-readable name for a given algorithm
    /// identifier.</remarks>
    public interface IAlgorithmIdentity {
        public const string MatrixFactorizationDisplayName = @"Matrix Factorization";
        public const string LBFGSOptimizationDisplayName = @"L-BFGS Optimization";
        public const string KMeansClusteringDisplayName = @"K-Means++";
        public const string LightGbmDisplayName = @"LightGBM";
        public const string FastTreeRegressionDisplayName = @"Fast Tree Regression";
        public const string FastForestRegressionDisplayName = @"Fast Forest Regression";
        public const string SdcaRegressionDisplayName = @"SDCA Regression";
        public const string LbfgsRegressionDisplayName = @"Limited-memory BFGS";

        /// <summary>
        /// Specifies the set of supported machine learning algorithms.
        /// </summary>
        /// <remarks>Use this enumeration to select the algorithm to be applied in machine learning
        /// workflows. The available values represent common algorithm types, such as matrix factorization, LBFGS
        /// optimization, k-means clustering, and forecasting. Not all algorithms may be implemented or available in
        /// every context; refer to the specific API documentation for details on supported operations.</remarks>
        public enum Algorithm {
            MatrixFactorization,
            LBFGSOptimization,
            KMeansClustering,
            LightGbm,
            FastTreeRegression,
            FastForestRegression,
            SdcaRegression,
            LbfgsRegression
        }

        /// <summary>
        /// Specifies the types of jobs that can be executed in the system.
        /// </summary>
        public enum WorkType {
            Recommendations,
            CustomerIntelligence, // Churn prediction + segmentation have been merged into this job type
            Forecasting,
            Waiting,
            AutoExperiment
        }

        /// <summary>
        /// Provides a mapping of supported algorithms to their display names.
        /// </summary>
        /// <remarks>The dictionary contains the set of algorithms recognized by the system and their
        /// corresponding human-readable names. The set of supported algorithms is fixed at initialization and does not
        /// change at runtime.</remarks>
        public static readonly Dictionary<Algorithm, string> SupportedAlgorithms = new() {
            { Algorithm.MatrixFactorization, MatrixFactorizationDisplayName },
            { Algorithm.LBFGSOptimization, LBFGSOptimizationDisplayName },
            { Algorithm.KMeansClustering, KMeansClusteringDisplayName },
            { Algorithm.LightGbm, LightGbmDisplayName }
        };

        /// <summary>
        /// Provides a mapping of algorithm display names to their corresponding <see cref="Algorithm"/> enumeration
        /// values.
        /// </summary>
        /// <remarks>This dictionary can be used to look up supported algorithms by their human-readable
        /// names. The set of supported algorithms is fixed and reflects the algorithms available in the current
        /// version.</remarks>
        public static readonly Dictionary<string, Algorithm> AlgorithmSupported = new() {
            { MatrixFactorizationDisplayName, Algorithm.MatrixFactorization },
            { LBFGSOptimizationDisplayName, Algorithm.LBFGSOptimization },
            { KMeansClusteringDisplayName, Algorithm.KMeansClustering },
            { LightGbmDisplayName, Algorithm.LightGbm },
            { FastTreeRegressionDisplayName, Algorithm.FastTreeRegression },
            { FastForestRegressionDisplayName, Algorithm.FastForestRegression },
            { SdcaRegressionDisplayName, Algorithm.SdcaRegression },
            { LbfgsRegressionDisplayName, Algorithm.LbfgsRegression }
        };
    }
}
