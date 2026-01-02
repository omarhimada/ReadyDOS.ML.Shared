using Microsoft.ML.Data;

/// <summary>
/// Defines the contract for a workflow metric that captures evaluation results and performance statistics for a machine
/// learning model run.
/// </summary>
/// <remarks>Implementations of this interface provide access to common regression and classification metrics,
/// such as R-squared, RMSE, accuracy, and AUC, as well as metadata identifying the model and source file. All metric
/// properties are nullable to accommodate scenarios where certain metrics may not be applicable or available for a
/// given model type.</remarks>
public interface IWorkflowMetric
{
    Guid Id { get; set; }
    string ModelName { get; set; }
    string FileName { get; set; }
    double? RSquared { get; set; }
    double? RMSE { get; set; }
    double? MeanAbsoluteError { get; set; }
    double? MeanSquaredError { get; set; }
    double? LossFunction { get; set; }
    double? AUC { get; set; }
    double? Accuracy { get; set; }
    double? F1Score { get; set; }
    double? Precision { get; set; }
    double? Recall { get; set; }
    double? LogLoss { get; set; }
    double? LogLossReduction { get; set; }
    public bool? Binary { get; set; }
    public MulticlassClassificationMetrics? Multiclass { get; set; }
}
