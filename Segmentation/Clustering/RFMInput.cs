namespace ReadyDOS.Shared.Segmentation.Clustering {

    /// <summary>
    /// Represents input data for customer segmentation analysis, including recency, frequency, and monetary value
    /// metrics. You can extract this from your customer database or CRM system, a spreadsheet, or an API. 
    /// It is the beginning input model. Later on, prioritization of each segment and a customizable display name is added.
    /// </summary>
    /// <remarks>This type is typically used to provide features such as recency (in days), frequency, and
    /// monetary value for algorithms that segment customers based on their behavior. The values should be set according
    /// to the requirements of the segmentation model being used.</remarks>
    public sealed class RFMInput {
        // Replace with your real features (RFM, usage, spend, etc.)
        public float RecencyDays { get; set; }
        public float Frequency { get; set; }
        public float Monetary { get; set; }
    }
}
