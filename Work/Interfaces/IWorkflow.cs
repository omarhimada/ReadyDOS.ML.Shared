namespace ReadyDOS.Shared.Work.Interfaces {
    using Microsoft.Extensions.Logging;
    using static Shared.Work.Interfaces.IAlgorithmIdentity;

    /// <summary>
    /// Defines the contract for a workflow that manages the execution of jobs, data splits, and algorithm runs,
    /// providing methods and data structures for orchestrating and tracking workflow operations.
    /// </summary>
    /// <remarks>The IWorkflow interface provides a set of records and methods for representing
    /// workflow-related metadata, configuring and executing workflow runs, and capturing execution results and logs.
    /// Implementations are expected to support asynchronous execution and provide detailed tracking of workflow
    /// progress and outcomes. All time-related values are expressed in Coordinated Universal Time (UTC) to ensure
    /// consistency across distributed systems.</remarks>
    public interface IWorkflow {
        public const string MatrixFactorizationDisplayName = @"Matrix Factorization";
        public const string LBFGSOptimizationDisplayName = @"L-BFGS Optimization";
        public const string KMeansClusteringDisplayName = @"K-Means++";
        public const string LightGbmRegressionDisplayName = @"LightGBM Regression";
        public const string LightGbmDisplayName = @"LightGBM";
        public const string FastTreeRegressionDisplayName = @"Fast Tree Regression";
        public const string FastForestRegressionDisplayName = @"Fast Forest Regression";
        public const string SdcaRegressionDisplayName = @"SDCA Regression";
        public const string LbfgsRegressionDisplayName = @"Limited-memory BFGS";

        public const string trainerNameMatrixFactorization = "MatrixFactorization";
        public const string trainerNameSdcaRegression = "SdcaRegression";
        public const string trainerNameLbfgsPoissonRegression = "LbfgsPoissonRegression";
        public const string trainerNameFastForestRegression = "FastForestRegression";
        public const string trainerNameFastTreeRegression = "FastTreeRegression";
        public const string trainerNameLightGbmRegression = "LightGbmRegression";
        public const string trainerNameLightGbm = "LightGbm";

        public static Dictionary<string, string> TrainerNameToDisplayName = new() {
            {  trainerNameSdcaRegression, SdcaRegressionDisplayName },
            {  trainerNameLbfgsPoissonRegression, LbfgsRegressionDisplayName },
            {  trainerNameFastForestRegression, FastForestRegressionDisplayName },
            {  trainerNameFastTreeRegression, FastTreeRegressionDisplayName },
            {  trainerNameLightGbmRegression, LightGbmRegressionDisplayName },
            {  trainerNameLightGbm, LightGbmDisplayName },
            {  trainerNameMatrixFactorization, MatrixFactorizationDisplayName }
        };

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
            { Algorithm.LightGbm, LightGbmDisplayName },
            { Algorithm.FastTreeRegression, FastTreeRegressionDisplayName },
            { Algorithm.FastForestRegression, FastForestRegressionDisplayName },
            { Algorithm.SdcaRegression, SdcaRegressionDisplayName },
            { Algorithm.LbfgsRegression, LbfgsRegressionDisplayName }
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