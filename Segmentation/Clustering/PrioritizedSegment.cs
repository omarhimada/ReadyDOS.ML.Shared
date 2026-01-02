namespace ReadyDOS.Shared.Segmentation.Clustering {

    /// <summary>
    /// Represents a customer segment that has been prioritized for actionable business response.
    /// Characterized by recency, frequency, and monetary metrics, commonly used in RFM, 
    /// (Recency, Frequency, Monetary) analysis for marketing and customer targeting. 
    /// Distance to centroid and business implication:
    /// Low(e.g., 0.34 – Segment 5) -> very tight, very similar customers, Safe to automate, reliable for targeted campaigns
    /// High(e.g., 1.13 – Segment 2) -> Loose, mixed behavior   Flag it for analysis, test carefully, or re-segment
    /// Includes the normalized values for each metric along with the raw values for display purposes.
    /// This makes it adaptable to many datasets while retaining interpretability.
    /// </summary>
    /// <remarks>
    /// Use this class to model and analyze customer behavior for segmentation purposes, such as
    /// identifying high-value or recently active customers. The properties correspond to standard RFM metrics, which
    /// can be used to prioritize marketing efforts or personalize customer outreach.
    /// </remarks>
    public class PrioritizedSegment {
        /// <summary>
        /// Gets or sets the display name associated with the object. 
        /// Due to normalization, these names are consistent across datasets.
        /// </summary>
        public string DisplayName { get; set; } = string.Empty;
        public double Recency { get; set; }
        public double Frequency { get; set; }
        public double Monetary { get; set; }

        /// <summary>
        /// Gets or sets the priority level for segment, in ascending order (1 = highest priority).
        /// (e.g.: Segment with Priority 1 is considered most important for a targeted marketing campaign.)
        /// </summary>
        public int Priority { get; set; }

        /// <summary>
        /// The number of customers/users in the segment.
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the set of values intended for display purposes. For example, generating
        /// a chart using a UI library for a dashboard or administrative interface.
        /// </summary>
        public required ForDisplayPurposes DisplayValues { get; set; }

        /// <summary>
        /// Represents a set of raw RFM (Recency, Frequency, Monetary) values for display or further processing.
        /// </summary>
        /// <remarks>This class is typically used to hold unprocessed RFM metrics prior to normalization,
        /// calibration, or correction. The values correspond to the original measurements and may require additional
        /// transformation before use in analytics or reporting.</remarks>
        public class ForDisplayPurposes {
            /// <summary>
            /// Gets or sets the raw recency value, measured in days.
            /// </summary>
            public double RawRecencyDays { get; set; }

            /// <summary>
            /// Gets or sets the unprocessed frequency value before any calibration or correction is applied.
            /// </summary>
            public double RawFrequency { get; set; }

            /// <summary>
            /// Gets or sets the raw monetary value before any normalization is applied.
            /// </summary>
            public double RawMonetary { get; set; }
        }
    }
}
