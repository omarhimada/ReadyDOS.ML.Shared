namespace ReadyDOS.Shared.Work {
    /// <summary>
    /// Represents a message containing information about the progress and identification of a workflow run.
    /// </summary>
    public class WorkflowRunMessage {
        /// <summary>
        /// Gets or sets the estimated percentage value, if available.
        /// </summary>
        public double? EstimatedPercentage { get; set; }

        /// <summary>
        /// Gets or sets the number of steps completed in the process, or null if the step count is unknown.
        /// </summary>
        public int? StepCount { get; set; }

        /// <summary>
        /// Gets or sets the unique identifier for this instance.
        /// </summary>
        public Guid Identification { get; set; }

        /// <summary>
        /// Gets a user-friendly string representation of the identifier, removing hyphens.
        /// Consistency across integrations with this is important, as an example when using this as a part
        /// of a blob key.
        /// </summary>
        public string FriendlyIdentifier => $"{Identification:N}";
    }
}
