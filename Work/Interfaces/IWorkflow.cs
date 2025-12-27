namespace Shared.Work.Interfaces {
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

        /// <summary>
        /// Begins a new iteration of the specified job type using the provided algorithm identity.
        /// </summary>
        /// <typeparam name="T">The type of algorithm identity to use for the iteration. Must inherit from AlgorithmIdentity.</typeparam>
        /// <param name="jobType">The type of job to execute for this iteration.</param>
        /// <param name="stoppingToken">A cancellation token that can be used to request cancellation of the iteration.</param>
        /// <returns>A task that represents the asynchronous operation of starting the iteration.</returns>
        public Task<WorkflowLogMessage> BeginIteration(JobType jobType, CancellationToken stoppingToken = default);

        /// <summary>
        /// Executes the workflow asynchronously using the specified algorithm and run options.
        /// </summary>
        /// <param name="options">The options that configure the workflow run, including algorithm parameters and execution settings. Cannot
        /// be null.</param>
        /// <returns>A task that represents the asynchronous operation. The result contains information about the workflow run,
        /// including status and output data.</returns>
        Task<WorkflowRunResult> RunAsync(WorkflowRunOptions options, CancellationToken stoppingToken = default);

        /// <summary>
        /// Represents information about a data split, including references to training and evaluation datasets and
        /// associated metadata.
        /// </summary>
        /// <param name="trainingData">The dataset used for training. This object should contain the data intended for model training. Cannot be
        /// null.</param>
        /// <param name="evaluationData">The dataset used for evaluation. This object should contain the data reserved for model evaluation. Cannot
        /// be null.</param>
        /// <param name="sourceKey">A string that identifies the source or origin of the data split. Cannot be null or empty.</param>
        /// <param name="trainRowCount">The number of rows in the training dataset, if known. Specify null if the row count is not available.</param>
        /// <param name="evalRowCount">The number of rows in the evaluation dataset, if known. Specify null if the row count is not available.</param>
        sealed record DataSplitInfo() {
            public required object TrainingData { get; init; }
            public required object EvaluationData { get; init; }
            public required string SourceKey { get; init; }
            public long? TrainRowCount { get; init; }
            public long? EvalRowCount { get; init; }
        }

        /// <summary>
        /// Represents metadata information for a data object stored in Amazon S3, or Azure Blob Storage, including its S3 key and the last
        /// modified timestamp.
        /// </summary>
        /// <param name="Key">The unique key identifying the object in the S3 bucket, or Azure Blob Storage. Cannot be null or empty.</param>
        /// <param name="LastModifiedUtc">The date and time, in Coordinated Universal Time (UTC), when the object was last modified, or null if the
        /// modification time is unknown.</param>
        sealed record DataObjectInfo() {
            public required string Key { get; init; }
            public DateTime? LastModifiedUtc { get; init; }
        }

        /// <summary>
        /// Specifies options for configuring the execution of a workflow run.
        /// </summary>
        /// <param name="jobType">The type of job to execute within the workflow. Determines the workflow's behavior and processing logic.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the workflow run. If not specified, the workflow will run until
        /// completion unless cancelled by other means.</param>
        public sealed record WorkflowRunOptions() {
            public JobType JobType { get; init; }
            public CancellationToken CancellationToken { get; init; }
        }

        
        /// <summary>
        /// Represents the result of a workflow run, including execution details, status, and summary information.
        /// </summary>
        /// <remarks>This record provides a snapshot of a completed workflow execution, including timing,
        /// outcome, and optional diagnostic information. It is typically used to report or analyze the outcome of a
        /// workflow job after execution has finished.</remarks>
        public sealed record WorkflowRunResult { 
            public JobType JobType { get; init; }
            public required string AlgorithmDisplayName { get; init; }
            public string? SourceKey { get; init; }
            public DateTime StartedAtUtc { get; init; }
            public DateTime FinishedAtUtc { get; init; }
            public bool Succeeded { get; init; }
            public required string Summary { get; init; }
            public IReadOnlyList<WorkflowLogMessage>? Logs { get; init; }
            public object? MetricSummary { get; init; }
        }

        /// <summary>
        /// Represents a log message generated during workflow execution, including category, message text, timestamp,
        /// and optional event metadata.
        /// </summary>
        /// <remarks>This record is typically used to capture and transport workflow-related logging
        /// information for diagnostics, auditing, or monitoring purposes. All timestamps are in UTC to ensure
        /// consistency across distributed systems.</remarks>
        /// <param name="Category">The category or source of the log message, used to group related workflow events. Cannot be null.</param>
        /// <param name="Message">The descriptive text of the log message. Cannot be null.</param>
        /// <param name="TimestampUtc">The UTC timestamp indicating when the log message was created.</param>
        /// <param name="EventId">An optional identifier for the workflow event associated with this log message. If not specified, the
        /// message may not be linked to a specific event.</param>
        /// <param name="EventName">An optional name for the workflow event associated with this log message. If not specified, the message may
        /// not be linked to a named event.</param>
        /// <param name="Logger">An optional logger instance that produced this message. If null, the message may have been generated
        /// independently of a specific logger.</param>
        public record WorkflowLogMessage {
            public required string Category { get; init; }
            public required string Message { get; init; }
            public DateTime TimestampUtc { get; init; }
            public EventId? EventId { get; init; }
            public string? EventName { get; init; }
            public ILogger? Logger { get; init; }
        }
    }
}