using Microsoft.ML.Data;
namespace ReadyDOS.Shared.Work.Interfaces {
    /// <summary>
    /// Defines the contract for a workflow metric that captures evaluation results and performance statistics for a machine
    /// learning model run.
    /// </summary>
    /// <remarks>Implementations of this interface provide access to common regression and classification metrics,
    /// such as R-squared, RMSE, accuracy, and AUC, as well as metadata identifying the model and source file. All metric
    /// properties are nullable to accommodate scenarios where certain metrics may not be applicable or available for a
    /// given model type.</remarks>
    public interface IWorkflowMetric {
        /// <summary>
        /// Gets or sets the unique identifier for the workflow.
        /// </summary>
        Guid Id { get; set; }
        /// <summary>
        /// Gets or sets the name of the model associated with the workflow instance.
        /// </summary>
        string ModelName { get; set; }
        /// <summary>
        /// Gets or sets the name of the file associated with the workflow instance.
        /// </summary>
        string FileName { get; set; }

        /// <summary>
        /// Gets or sets the metrics for evaluating multiclass classification performance.
        /// </summary>
        public MulticlassClassificationMetrics? Multiclass { get; set; }

        /// <summary>
        /// Gets or sets the coefficient of determination (R-squared) for the model.
        /// </summary>
        /// <remarks>R-squared indicates how well the model explains the variability of the response data. A value
        /// closer to 1.0 suggests a better fit, while a value closer to 0 indicates a poor fit. The value is null if the
        /// R-squared statistic is not available or has not been computed.</remarks>
        double? RSquared { get; set; }
        /// <summary>
        /// Gets or sets the root mean square error (RMSE) value for the model's predictions.
        /// </summary>
        double? RMSE { get; set; }

        /// <summary>
        /// Gets or sets the mean absolute error of the model's predictions.
        /// </summary>
        double? MeanAbsoluteError { get; set; }
        /// <summary>
        /// Gets or sets the mean squared error value for the model's predictions.
        /// </summary>
        double? MeanSquaredError { get; set; }
        /// <summary>
        /// Gets or sets the loss function value used to evaluate model performance.
        /// </summary>
        double? LossFunction { get; set; }
        /// <summary>
        /// Gets or sets the area under the receiver operating characteristic (ROC) curve (AUC) for the model's predictions.
        /// </summary>
        /// <remarks>A higher AUC value indicates better model performance in distinguishing between positive and
        /// negative classes. The value is typically in the range [0, 1], where 1 represents perfect discrimination and 0.5
        /// represents random guessing.</remarks>
        double? AUC { get; set; }
        /// <summary>
        /// Gets or sets the accuracy value associated with the result, if available.
        /// </summary>
        double? Accuracy { get; set; }
        /// <summary>
        /// Gets or sets the F1 score, which represents the harmonic mean of precision and recall for a classification
        /// model.
        /// </summary>
        /// <remarks>The F1 score ranges from 0 to 1, where higher values indicate better balance between
        /// precision and recall. A value of <see langword="null"/> indicates that the F1 score is not available or has not
        /// been calculated.</remarks>
        double? F1Score { get; set; }
        /// <summary>
        /// Gets or sets the number of decimal places to use when displaying or processing values.
        /// </summary>
        double? Precision { get; set; }
        /// <summary>
        /// Gets or sets the recall metric value for the evaluation result.
        /// </summary>
        /// <remarks>Recall measures the proportion of relevant items that were correctly identified. The value is
        /// typically between 0.0 and 1.0, where higher values indicate better performance. A null value indicates that
        /// recall is not available or has not been calculated.</remarks>
        double? Recall { get; set; }
        /// <summary>
        /// Gets or sets the logarithmic loss (log loss) value for the model evaluation.
        /// </summary>
        /// <remarks>Log loss measures the performance of a classification model where the prediction input is a
        /// probability value between 0 and 1. Lower values indicate better model performance. A value of <see
        /// langword="null"/> indicates that the log loss is not available or has not been computed.</remarks>
        double? LogLoss { get; set; }
        /// <summary>
        /// Gets or sets the reduction in log loss achieved by the model compared to a baseline predictor.
        /// </summary>
        /// <remarks>A higher value indicates better model performance relative to the baseline. The value is
        /// typically calculated as the difference between the log loss of the baseline and the log loss of the model, and
        /// may be null if not computed or not applicable.</remarks>
        double? LogLossReduction { get; set; }
        /// <summary>
        /// Whether or not it is a Binary Classification model.
        /// </summary>
        public bool? Binary { get; set; }
        /// <summary>
        /// Describes when the workrflow began training.
        /// </summary>
        public DateTime TrainedAtUtc { get; set; }
    }
}
