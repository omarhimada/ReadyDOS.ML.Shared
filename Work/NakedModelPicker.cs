namespace Shared.Work {
    internal class NakedModelPicker {
        public enum ScoreSource { Auc, RSquared, NegRmse, NegLogLoss, MicroAccuracy, Unknown }

        public sealed record RankedModel(
            IWorkflowMetric Metric,
            double Score,
            ScoreSource Source,
            string Detail
        );

        /// <summary>
        /// Provides functionality to select and rank workflow models based on their evaluation metrics, enabling
        /// identification of the best-performing model from a collection of candidates.
        /// </summary>
        /// <remarks>The ModelPicker class evaluates each workflow metric using a prioritized set of
        /// scoring criteria, including AUC, R-squared, micro-accuracy, RMSE, and LogLoss. It ensures that higher scores
        /// always indicate better performance, regardless of the original metric's directionality. This class is
        /// intended for scenarios where multiple models or workflows are evaluated and a single best candidate must be
        /// selected based on standardized ranking logic.</remarks>
        public static class ModelPicker {
            public static RankedModel? PickBest(IEnumerable<IWorkflowMetric> items) {
                var ranked = items
                    .Select(TryRank)
                    .Where(x => x is not null)
                    .Select(x => x!)
                    // Higher is better (we negate error metrics below)
                    .OrderByDescending(x => x.Score)
                    // Tie-break: most recent completion
                    .ThenByDescending(x => x.Metric)
                    .FirstOrDefault();

                return ranked;
            }

            /// <summary>
            /// Attempts to create a ranked representation of the specified workflow metric using the most appropriate
            /// available scoring metric.
            /// </summary>
            /// <remarks>The method prioritizes metrics in the following order: binary AUC, regression
            /// R-squared, multiclass micro-accuracy, RMSE (negated), and LogLoss (negated). Only the first available
            /// and finite metric is used for ranking. Lower-is-better metrics are negated to ensure higher scores
            /// indicate better performance.</remarks>
            /// <param name="m">The workflow metric to evaluate and rank. Must provide at least one supported metric value.</param>
            /// <returns>A RankedModel instance representing the ranking of the workflow metric if a suitable metric is available
            /// and valid; otherwise, null.</returns>
            private static RankedModel? TryRank(IWorkflowMetric m) {
                // 1) Binary AUC if present
                double? auc = m.AUC;
                if (auc is double a && IsFinite(a)) return 
                        new RankedModel(m, a, ScoreSource.Auc, $"AUC={a:0.####}");

                // 2) Regression R²
                double? r2 = m.RSquared;
                if (r2 is double r && IsFinite(r)) return 
                        new RankedModel(m, r, ScoreSource.RSquared, $"R2={r:0.####}");

                // 3) Multiclass micro-accuracy
                double? micro = m.Multiclass?.MicroAccuracy;
                if (micro is double mi && IsFinite(mi)) return 
                        new RankedModel(m, mi, ScoreSource.MicroAccuracy, $"MicroAcc={mi:0.####}");

                // 4) Lower-is-better metrics → negate
                double? rmse = m.RMSE;
                if (rmse is double e && IsFinite(e)) return 
                        new RankedModel(m, -e, ScoreSource.NegRmse, $"RMSE={e:0.####} (score=-RMSE)");

                double? logloss = m.LogLoss;
                if (logloss is double ll && IsFinite(ll)) return 
                        new RankedModel(m, -ll, ScoreSource.NegLogLoss, $"LogLoss={ll:0.####} (score=-LogLoss)");

                return null;
            }

            private static bool IsFinite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);
        }
    }
}
