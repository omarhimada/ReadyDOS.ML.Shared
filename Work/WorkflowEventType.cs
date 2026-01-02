namespace ReadyDOS.Shared.Work {
    /// <summary>
    /// Specifies the types of events that can occur during the execution of a workflow.
    /// </summary>
    /// <remarks>Use this enumeration to identify and handle specific workflow events, such as when a workflow
    /// starts, completes, fails, or processes datasets. The values can be used for logging, event handling, or
    /// monitoring workflow progress.</remarks>
    public enum WorkflowEventType {
        Starting = 1,
        Configuring = 2,
        Completed = 3,
        Failed = 4,
        DatasetDownloaded = 5,
        DatasetSplit = 6,
        Training = 7,
        Evaluating = 8,
        PersistingModel = 9,
        DatasetException = 10,
    };
}
