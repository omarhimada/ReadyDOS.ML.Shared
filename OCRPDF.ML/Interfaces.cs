using System.Drawing;

namespace Shared.OCRPDF.ML {
    /// <summary>
    /// Defines an OCR engine capable of recognizing text from <see cref="Bitmap"/> images asynchronously.
    /// </summary>
    /// <remarks>
    /// Implementations may hold native resources and therefore must be disposed.
    /// Thread-safety is implementation-specific.
    /// </remarks>
    public interface IOcrEngine : IDisposable {
        /// <summary>
        /// Recognizes text from the provided bitmap image.
        /// </summary>
        /// <param name="bitmap">Bitmap containing the text to recognize.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Recognized text (never null).</returns>
        Task<string> RecognizeAsync(Bitmap bitmap, CancellationToken ct);
    }

    /// <summary>
    /// Defines a PDF rasterizer that can render PDF pages to <see cref="Bitmap"/> images for downstream OCR.
    /// </summary>
    /// <remarks>
    /// Implementations may depend on native PDF rendering backends and should be disposed when no longer needed.
    /// </remarks>
    public interface IPdfRasterizer : IDisposable {
        /// <summary>
        /// Gets the number of pages in the specified PDF document.
        /// </summary>
        /// <param name="pdfPath">Path to a PDF file.</param>
        /// <returns>The number of pages reported by the backend.</returns>
        int GetPageCount(string pdfPath);

        /// <summary>
        /// Renders a single PDF page into a bitmap at the specified DPI.
        /// </summary>
        /// <param name="pdfPath">Path to a PDF file.</param>
        /// <param name="pageIndex">Zero-based page index.</param>
        /// <param name="dpi">Target DPI for rendering.</param>
        /// <returns>Cluster rendered bitmap that must be disposed by the caller.</returns>
        Bitmap RenderPage(string pdfPath, int pageIndex, int dpi);
    }

    /// <summary>
    /// Defines an embedded-text extractor for PDF documents.
    /// </summary>
    public interface IPdfTextExtractor : IDisposable {
        /// <summary>
        /// Extracts all available embedded text from the specified PDF file.
        /// </summary>
        /// <param name="pdfPath">Path to a PDF file.</param>
        /// <returns>Concatenated extracted text (may be empty).</returns>
        string ExtractAllText(string pdfPath);
    }
}
