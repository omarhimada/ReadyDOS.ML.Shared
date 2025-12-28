namespace Shared.OCRPDF.ML {
    using global::Shared.Work.Interfaces;
    using Microsoft.Extensions.Logging;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Transforms.Text;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Security.Cryptography;
    using System.Text;
    using System.Text.Json;
    using System.Threading;
    using System.Threading.Tasks;

    namespace Shared {

        /// <summary>
        /// Workflow that ingests PDF documents, extracts text either from embedded PDF text or via OCR,
        /// featurizes the resulting text, and trains a clustering model (KMeans) using ML.NET.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This workflow supports two extraction paths:
        /// </para>
        /// <list type="bullet">
        /// <item><description><b>Embedded text extraction</b> via UglyToad.PdfPig (if present at runtime).</description></item>
        /// <item><description><b>OCR extraction</b> via Tesseract + PdfiumViewer rasterization (if present at runtime).</description></item>
        /// </list>
        /// <para>
        /// If both extraction paths are unavailable, the workflow fails with an <see cref="InvalidOperationException"/>.
        /// </para>
        /// <para>
        /// Artifacts are written to <see cref="PdfOcrTrainingOptions.OutputDirectory"/> (or an iteration-specific subfolder
        /// when invoked via <see cref="RunAsync(IWorkflow.WorkflowRunOptions, CancellationToken)"/>).
        /// </para>
        /// </remarks>
        public sealed class PdfOcrTrainingWorkflow : IWorkflow {
            private readonly ILogger _log;
            private readonly PdfOcrTrainingOptions _options;

            // -------------------------------------------------------------------------
            // Add these private fields somewhere in your workflow class
            // -------------------------------------------------------------------------
            private int _iterationNumber = 0;
            private string? _currentIterationId = null;
            private string? _lastModelPath = null;
            private string? _lastManifestPath = null;
            private ClusteringQuality? _lastQuality = null;

            /// <summary>
            /// Initializes a new instance of the <see cref="PdfOcrTrainingWorkflow"/> class.
            /// </summary>
            /// <param name="log">Logger used for workflow diagnostics and progress reporting.</param>
            /// <param name="options">Training options specifying input PDFs, OCR settings, clustering parameters, and artifact paths.</param>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="log"/> or <paramref name="options"/> is null.</exception>
            /// <exception cref="ArgumentException">Thrown when <paramref name="options"/> does not contain any PDF inputs.</exception>
            public PdfOcrTrainingWorkflow(ILogger log, PdfOcrTrainingOptions options) {
                _log = log ?? throw new ArgumentNullException(nameof(log));
                _options = options ?? throw new ArgumentNullException(nameof(options));

                if (_options.PdfFiles == null || _options.PdfFiles.Count == 0) {
                    throw new ArgumentException("Options must include at least one PDF file path.", nameof(options));
                }
            }

            /// <summary>
            /// Gets the workflow name as reported to the hosting system.
            /// </summary>
            /// <remarks>
            /// If <see cref="PdfOcrTrainingOptions.WorkflowName"/> is null, a default name is used.
            /// </remarks>
            public string Name => _options.WorkflowName ?? "PdfOcrTrainingWorkflow";

            /// <summary>
            /// Gets a short human-readable description of the workflow.
            /// </summary>
            public string? Description => "Trains a clustering model from PDF documents using embedded text or OCR.";

            /// <summary>
            /// Runs the workflow using the default execution pathway.
            /// </summary>
            /// <param name="ct">Cancellation token.</param>
            /// <returns>A task representing the asynchronous execution.</returns>
            public Task RunAsync(CancellationToken ct = default) => ExecuteCoreAsync(ct);

            /// <summary>
            /// Executes the workflow using the default execution pathway.
            /// </summary>
            /// <param name="ct">Cancellation token.</param>
            /// <returns>A task representing the asynchronous execution.</returns>
            public Task ExecuteAsync(CancellationToken ct = default) => ExecuteCoreAsync(ct);

            /// <summary>
            /// Begins a workflow iteration and returns a structured log message suitable for persistence.
            /// </summary>
            /// <remarks>
            /// This is designed to be called by timed/recurring hosts (for example, a <c>BackgroundService</c>)
            /// to mark an iteration boundary and associate artifacts with an iteration id.
            /// </remarks>
            /// <param name="jobType">The job type associated with this iteration (if provided by the host).</param>
            /// <param name="stoppingToken">Cancellation token.</param>
            /// <returns>A workflow log message describing the iteration start.</returns>
            public async Task<IWorkflow.WorkflowLogMessage> BeginIteration(IAlgorithmIdentity.JobType jobType, CancellationToken stoppingToken = default) {
                stoppingToken.ThrowIfCancellationRequested();

                int next = Interlocked.Increment(ref _iterationNumber);
                string iterationId = DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture) + "-i" + next.ToString(CultureInfo.InvariantCulture);

                _currentIterationId = iterationId;

                string message = $"[{Name}] BeginIteration: jobType={jobType}, iteration={next}, iterationId={iterationId}, utc={DateTimeOffset.UtcNow:O}";
                _log.LogInformation(message);

                IWorkflow.WorkflowLogMessage logMessage = CreateObject<IWorkflow.WorkflowLogMessage>();
                SetIfExists(logMessage, "Message", message);
                SetIfExists(logMessage, "Text", message);
                SetIfExists(logMessage, "Level", "Information");
                SetIfExists(logMessage, "Severity", "Information");
                SetIfExists(logMessage, "TimestampUtc", DateTimeOffset.UtcNow);
                SetIfExists(logMessage, "Timestamp", DateTimeOffset.UtcNow);
                SetIfExists(logMessage, "JobType", jobType);
                SetIfExists(logMessage, "IterationId", iterationId);
                SetIfExists(logMessage, "IterationNumber", next);

                return await Task.FromResult(logMessage).ConfigureAwait(false);
            }

            /// <summary>
            /// Runs the workflow in an “iteration-aware” mode, producing per-iteration artifacts and optionally promoting
            /// improved candidates to a <c>best</c> directory.
            /// </summary>
            /// <remarks>
            /// <para>
            /// This method is intended to support automated retraining loops. It:
            /// </para>
            /// <list type="number">
            /// <item><description>Creates a new iteration id and an iteration-specific artifacts directory.</description></item>
            /// <item><description>Executes training via <see cref="ExecuteAsync(CancellationToken)"/>.</description></item>
            /// <item><description>Best-effort discovers iteration artifacts (model zip + manifest json).</description></item>
            /// <item><description>Optionally compares clustering quality against the last promoted model and promotes if improved.</description></item>
            /// </list>
            /// <para>
            /// “Best” promotion prefers a lower Davies–Bouldin Index when available, otherwise a lower average distance.
            /// If quality cannot be determined, the candidate is treated as promotable.
            /// </para>
            /// </remarks>
            /// <param name="options">Run options provided by the host (may include artifact directories or job type fields).</param>
            /// <param name="stoppingToken">Cancellation token.</param>
            /// <returns>A workflow run result describing success/failure and artifact paths.</returns>
            public async Task<IWorkflow.WorkflowRunResult> RunAsync(IWorkflow.WorkflowRunOptions options, CancellationToken stoppingToken = default) {
                stoppingToken.ThrowIfCancellationRequested();

                Stopwatch sw = Stopwatch.StartNew();

                IAlgorithmIdentity.JobType jobType = ReadIfExists<IAlgorithmIdentity.JobType>(options, "JobType", default(IAlgorithmIdentity.JobType));
                _ = await BeginIteration(jobType, stoppingToken).ConfigureAwait(false);

                string baseArtifactsDir = ResolveArtifactsDirectory(options);
                Directory.CreateDirectory(baseArtifactsDir);

                string iterationDir = Path.Combine(baseArtifactsDir, _currentIterationId ?? ("iter-" + Guid.NewGuid().ToString("N")));
                Directory.CreateDirectory(iterationDir);

                TrySetOptionsOutputDirectory(iterationDir);

                try {
                    await ExecuteAsync(stoppingToken).ConfigureAwait(false);
                }
                catch (Exception ex) {
                    sw.Stop();
                    string failMessage = $"[{Name}] RunAsync failed during ExecuteAsync. Iteration={_currentIterationId}. Error={ex.GetType().Name}: {ex.Message}";
                    _log.LogError(failMessage, ex);

                    IWorkflow.WorkflowRunResult failed = CreateObject<IWorkflow.WorkflowRunResult>();
                    SetIfExists(failed, "Success", false);
                    SetIfExists(failed, "Succeeded", false);
                    SetIfExists(failed, "Message", failMessage);
                    SetIfExists(failed, "Error", ex.ToString());
                    SetIfExists(failed, "DurationMs", sw.ElapsedMilliseconds);
                    SetIfExists(failed, "IterationId", _currentIterationId);
                    SetIfExists(failed, "ArtifactsDirectory", iterationDir);

                    return failed;
                }

                if (string.IsNullOrWhiteSpace(_lastModelPath)) {
                    string? foundModel = FindNewestFile(iterationDir, "*.zip");
                    _lastModelPath = foundModel;
                }
                if (string.IsNullOrWhiteSpace(_lastManifestPath)) {
                    string? foundManifest = FindNewestFile(iterationDir, "*manifest*.json") ?? FindNewestFile(iterationDir, "*.json");
                    _lastManifestPath = foundManifest;
                }

                bool improved = true;
                ClusteringQuality? candidateQuality = null;

                try {
                    candidateQuality = TryReadClusteringQualityFromManifest(_lastManifestPath);
                    if (candidateQuality != null && _lastQuality != null) {
                        improved = candidateQuality.IsBetterThan(_lastQuality);
                    }
                }
                catch (Exception ex) {
                    _log.LogWarning($"[{Name}] Could not compare quality; defaulting to keep candidate. {ex.GetType().Name}: {ex.Message}");
                    improved = true;
                }

                string bestDir = Path.Combine(baseArtifactsDir, "best");
                Directory.CreateDirectory(bestDir);

                string bestModelPath = Path.Combine(bestDir, "model.zip");
                string bestManifestPath = Path.Combine(bestDir, "manifest.json");

                if (improved) {
                    if (!string.IsNullOrWhiteSpace(_lastModelPath) && File.Exists(_lastModelPath)) {
                        File.Copy(_lastModelPath, bestModelPath, overwrite: true);
                    }
                    if (!string.IsNullOrWhiteSpace(_lastManifestPath) && File.Exists(_lastManifestPath)) {
                        File.Copy(_lastManifestPath, bestManifestPath, overwrite: true);
                    }

                    if (candidateQuality != null) {
                        _lastQuality = candidateQuality;
                    }

                    _log.LogInformation($"[{Name}] Candidate improved -> promoted to BEST. Iteration={_currentIterationId}");
                }
                else {
                    _log.LogInformation($"[{Name}] Candidate NOT improved -> not promoted. Iteration={_currentIterationId}");
                }

                sw.Stop();

                string okMessage = improved
                    ? $"[{Name}] Run completed. Candidate improved and promoted. Iteration={_currentIterationId}"
                    : $"[{Name}] Run completed. Candidate did not improve; kept for audit only. Iteration={_currentIterationId}";

                IWorkflow.WorkflowRunResult result = CreateObject<IWorkflow.WorkflowRunResult>();
                SetIfExists(result, "Success", true);
                SetIfExists(result, "Succeeded", true);
                SetIfExists(result, "Improved", improved);
                SetIfExists(result, "Message", okMessage);
                SetIfExists(result, "DurationMs", sw.ElapsedMilliseconds);
                SetIfExists(result, "IterationId", _currentIterationId);
                SetIfExists(result, "ArtifactsDirectory", iterationDir);
                SetIfExists(result, "BestArtifactsDirectory", bestDir);
                SetIfExists(result, "ModelPath", improved ? bestModelPath : _lastModelPath);
                SetIfExists(result, "ManifestPath", improved ? bestManifestPath : _lastManifestPath);

                List<string> artifactPaths = new List<string>();
                if (!string.IsNullOrWhiteSpace(_lastModelPath))
                    artifactPaths.Add(_lastModelPath);
                if (!string.IsNullOrWhiteSpace(_lastManifestPath))
                    artifactPaths.Add(_lastManifestPath);
                if (File.Exists(bestModelPath))
                    artifactPaths.Add(bestModelPath);
                if (File.Exists(bestManifestPath))
                    artifactPaths.Add(bestManifestPath);
                SetIfExists(result, "ArtifactPaths", artifactPaths);

                return result;
            }

            /// <summary>
            /// Core implementation for ingestion, featurization, training, preview scoring, and artifact persistence.
            /// </summary>
            /// <param name="ct">Cancellation token.</param>
            /// <returns>A task representing the asynchronous operation.</returns>
            /// <exception cref="InvalidOperationException">
            /// Thrown when no text extraction pathway is available, or when no usable documents are ingested.
            /// </exception>
            private async Task ExecuteCoreAsync(CancellationToken ct) {
                Stopwatch sw = Stopwatch.StartNew();

                _log.LogInformation($"[{Name}] Starting PDF OCR training workflow.");
                _log.LogInformation($"[{Name}] PDF inputs: {_options.PdfFiles.Count}");
                _log.LogInformation($"[{Name}] Output directory: {_options.OutputDirectory}");

                Directory.CreateDirectory(_options.OutputDirectory);

                using IOcrEngine? ocr = CreateOcrEngineOrNull(_options);
                using IPdfRasterizer? rasterizer = CreatePdfRasterizerOrNull();
                using IPdfTextExtractor? textExtractor = CreatePdfTextExtractorOrNull();

                if (ocr == null && textExtractor == null) {
                    throw new InvalidOperationException(
                        "No text extraction path available. Install either:\n" +
                        "  - UglyToad.PdfPig (embedded text extraction), or\n" +
                        "  - Tesseract + PdfiumViewer (OCR via PDF rasterization)\n" +
                        "Then re-run.");
                }

                List<DocumentTextRow> docs = new List<DocumentTextRow>(_options.PdfFiles.Count);
                List<DocumentIngestStats> perDocStats = new List<DocumentIngestStats>(_options.PdfFiles.Count);

                foreach (string pdfPathRaw in _options.PdfFiles) {
                    ct.ThrowIfCancellationRequested();

                    if (string.IsNullOrWhiteSpace(pdfPathRaw)) {
                        continue;
                    }

                    string fullPath = Path.GetFullPath(pdfPathRaw);
                    if (!File.Exists(fullPath)) {
                        _log.LogWarning($"[{Name}] Skipping missing file: {fullPath}");
                        continue;
                    }

                    _log.LogInformation($"[{Name}] Ingesting: {fullPath}");

                    Stopwatch ingestSw = Stopwatch.StartNew();

                    string? embeddedText = null;
                    if (textExtractor != null) {
                        try {
                            string extracted = textExtractor.ExtractAllText(fullPath);
                            extracted = NormalizeText(extracted);

                            if (!string.IsNullOrWhiteSpace(extracted) && extracted.Length >= _options.MinimumUsefulTextChars) {
                                embeddedText = extracted;
                                _log.LogInformation($"[{Name}] Embedded text extracted ({embeddedText.Length} chars).");
                            }
                        }
                        catch (Exception ex) {
                            _log.LogWarning($"[{Name}] Embedded text extraction failed: {ex.GetType().Name}: {ex.Message}");
                            embeddedText = null;
                        }
                    }

                    string? finalText = embeddedText;

                    int ocrPages = 0;
                    if (finalText == null) {
                        if (ocr == null || rasterizer == null) {
                            _log.LogWarning($"[{Name}] No OCR path available for {Path.GetFileName(fullPath)} (missing OCR engine or rasterizer).");
                        }
                        else {
                            StringBuilder sb = new StringBuilder(16_384);

                            int pageCount;
                            try {
                                pageCount = rasterizer.GetPageCount(fullPath);
                            }
                            catch (Exception ex) {
                                _log.LogWarning($"[{Name}] Failed reading page count: {ex.GetType().Name}: {ex.Message}");
                                pageCount = 0;
                            }

                            int maxPages = _options.MaxPagesPerPdf <= 0 ? int.MaxValue : _options.MaxPagesPerPdf;
                            int pagesToProcess = Math.Min(pageCount, maxPages);

                            _log.LogInformation($"[{Name}] OCR path engaged. Pages: {pagesToProcess}/{pageCount} (max = {_options.MaxPagesPerPdf}).");

                            for (int pageIndex = 0; pageIndex < pagesToProcess; pageIndex++) {
                                ct.ThrowIfCancellationRequested();

                                using Bitmap bmp = rasterizer.RenderPage(fullPath, pageIndex, _options.RenderDpi);
                                string pageText = await ocr.RecognizeAsync(bmp, ct).ConfigureAwait(false);
                                pageText = NormalizeText(pageText);

                                if (!string.IsNullOrWhiteSpace(pageText)) {
                                    sb.AppendLine(pageText);
                                    sb.AppendLine();
                                }

                                ocrPages++;
                            }

                            string candidate = NormalizeText(sb.ToString());
                            if (!string.IsNullOrWhiteSpace(candidate) && candidate.Length >= _options.MinimumUsefulTextChars) {
                                finalText = candidate;
                                _log.LogInformation($"[{Name}] OCR extracted ({finalText.Length} chars).");
                            }
                            else {
                                _log.LogWarning($"[{Name}] OCR produced insufficient text ({candidate.Length} chars).");
                                finalText = null;
                            }
                        }
                    }

                    if (finalText == null) {
                        _log.LogWarning($"[{Name}] No usable text extracted; skipping: {fullPath}");
                        DocumentIngestStats skippedStats = new DocumentIngestStats {
                            FilePath = fullPath,
                            UsedEmbeddedText = false,
                            UsedOcr = false,
                            OcrPages = ocrPages,
                            ExtractedChars = 0,
                            Hash = ComputeSha256Hex(fullPath),
                            DurationMs = ingestSw.ElapsedMilliseconds
                        };
                        perDocStats.Add(skippedStats);
                        continue;
                    }

                    finalText = TruncateIfNeeded(finalText, _options.MaxCharsPerDocument);

                    string hash = ComputeSha256Hex(fullPath);

                    DocumentTextRow row = new DocumentTextRow {
                        DocumentId = hash,
                        FileName = Path.GetFileName(fullPath),
                        Text = finalText
                    };
                    docs.Add(row);

                    DocumentIngestStats stats = new DocumentIngestStats {
                        FilePath = fullPath,
                        UsedEmbeddedText = embeddedText != null,
                        UsedOcr = embeddedText == null,
                        OcrPages = ocrPages,
                        ExtractedChars = finalText.Length,
                        Hash = hash,
                        DurationMs = ingestSw.ElapsedMilliseconds
                    };
                    perDocStats.Add(stats);

                    _log.LogInformation($"[{Name}] Ingested OK (chars={finalText.Length}, ms={ingestSw.ElapsedMilliseconds}).");
                }

                if (docs.Count == 0) {
                    throw new InvalidOperationException($"[{Name}] No usable documents were ingested. Nothing to train.");
                }

                MLContext ml = new MLContext(seed: _options.Seed);

                _log.LogInformation($"[{Name}] Training model on {docs.Count} documents.");

                IDataView data = ml.Data.LoadFromEnumerable(docs);

                TextFeaturizingEstimator.Options textOptions = new TextFeaturizingEstimator.Options {
                    KeepDiacritics = false,
                    KeepPunctuations = false,
                    KeepNumbers = true,
                    StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options {
                        Language = TextFeaturizingEstimator.Language.English
                    },
                    WordFeatureExtractor = new WordBagEstimator.Options {
                        NgramLength = _options.NgramLength,
                        UseAllLengths = true
                    },
                    CharFeatureExtractor = new WordBagEstimator.Options {
                        NgramLength = Math.Max(3, _options.NgramLength),
                        UseAllLengths = false
                    },
                    Norm = TextFeaturizingEstimator.NormFunction.L2
                };

                IEstimator<ITransformer> pipeline =
                    ml.Transforms.Text.FeaturizeText(
                            outputColumnName: "Features",
                            inputColumnName: nameof(DocumentTextRow.Text))
                    .Append(ml.Clustering.Trainers.KMeans(
                            featureColumnName: "Features",
                            numberOfClusters: _options.NumberOfClusters));

                ITransformer model = pipeline.Fit(data);

                IDataView scored = model.Transform(data);
                IEnumerable<ClusterPredictionRow> scoredRows = ml.Data.CreateEnumerable<ClusterPredictionRow>(scored, reuseRowObject: false);
                List<ClusterPredictionRow> preview = scoredRows.Take(_options.PreviewCount).ToList();

                foreach (ClusterPredictionRow p in preview) {
                    _log.LogInformation($"[{Name}] Preview -> {p.FileName} :: Cluster={p.PredictedLabel}, Score0={SafeFirst(p.Score)}");
                }

                string modelPath = Path.Combine(_options.OutputDirectory, _options.ModelFileName ?? "pdf-ocr-clustering-model.zip");
                using (FileStream fs = File.Create(modelPath)) {
                    ml.Model.Save(model, data.Schema, fs);
                }

                TrainingManifest manifest = new TrainingManifest {
                    WorkflowName = Name,
                    CreatedUtc = DateTimeOffset.UtcNow,
                    DocumentCount = docs.Count,
                    Options = _options,
                    ModelPath = modelPath,
                    Preview = preview,
                    IngestStats = perDocStats
                };

                string manifestPath = Path.Combine(_options.OutputDirectory, _options.ManifestFileName ?? "pdf-ocr-training-manifest.json");
                string manifestJson = JsonSerializer.Serialize(manifest, JsonOptions.Pretty);
                await File.WriteAllTextAsync(manifestPath, manifestJson, Encoding.UTF8, ct).ConfigureAwait(false);

                _log.LogInformation($"[{Name}] Model saved: {modelPath}");
                _log.LogInformation($"[{Name}] Manifest saved: {manifestPath}");
                _log.LogInformation($"[{Name}] Done in {sw.Elapsed.TotalSeconds:F1}s.");
            }

            /// <summary>
            /// Attempts to create an OCR engine instance using the specified training options, or returns null if the
            /// required OCR engine is unavailable.
            /// </summary>
            /// <remarks>
            /// This method attempts to locate and load the Tesseract OCR engine at runtime. If the Tesseract assembly
            /// is not found or cannot be loaded, the method returns null instead of throwing an exception.
            /// </remarks>
            /// <param name="options">The OCR training options to use when creating the engine.</param>
            /// <returns>An instance of an OCR engine if available; otherwise, null.</returns>
            private static IOcrEngine? CreateOcrEngineOrNull(PdfOcrTrainingOptions options) {
                try {
                    Assembly? tesseractAsm = AppDomain.CurrentDomain.GetAssemblies()
                        .FirstOrDefault(a => string.Equals(a.GetName().Name, "Tesseract", StringComparison.OrdinalIgnoreCase));

                    if (tesseractAsm == null) {
                        tesseractAsm = TryLoadAssembly("Tesseract");
                    }

                    if (tesseractAsm == null) {
                        return null;
                    }

                    return new TesseractOcrEngine(options);
                }
                catch {
                    return null;
                }
            }

            /// <summary>
            /// Attempts to create an instance of an <see cref="IPdfRasterizer"/> implementation using the PdfiumViewer
            /// library, if available.
            /// </summary>
            /// <remarks>
            /// This method returns null if the PdfiumViewer assembly is not found or if an error occurs during
            /// instantiation. Callers should check the return value for null before using the rasterizer.
            /// </remarks>
            /// <returns>An <see cref="IPdfRasterizer"/> instance if PdfiumViewer is present; otherwise, null.</returns>
            private static IPdfRasterizer? CreatePdfRasterizerOrNull() {
                try {
                    Assembly? asm = AppDomain.CurrentDomain.GetAssemblies()
                        .FirstOrDefault(a => string.Equals(a.GetName().Name, "PdfiumViewer", StringComparison.OrdinalIgnoreCase));

                    if (asm == null) {
                        asm = TryLoadAssembly("PdfiumViewer");
                    }

                    if (asm == null) {
                        return null;
                    }

                    return new PdfiumViewerRasterizer();
                }
                catch {
                    return null;
                }
            }

            /// <summary>
            /// Attempts to create an instance of an <see cref="IPdfTextExtractor"/> implementation using the
            /// UglyToad.PdfPig library, if available.
            /// </summary>
            /// <remarks>
            /// This method returns null if the UglyToad.PdfPig assembly is not found or if an error occurs during
            /// instantiation. Callers should check the return value for null before using the extractor.
            /// </remarks>
            /// <returns>An <see cref="IPdfTextExtractor"/> instance if PdfPig is present; otherwise, null.</returns>
            private static IPdfTextExtractor? CreatePdfTextExtractorOrNull() {
                try {
                    Assembly? asm = AppDomain.CurrentDomain.GetAssemblies()
                        .FirstOrDefault(a => string.Equals(a.GetName().Name, "UglyToad.PdfPig", StringComparison.OrdinalIgnoreCase));

                    if (asm == null) {
                        asm = TryLoadAssembly("UglyToad.PdfPig");
                    }

                    if (asm == null) {
                        return null;
                    }

                    return new PdfPigTextExtractor();
                }
                catch {
                    return null;
                }
            }

            /// <summary>
            /// Attempts to load an assembly by its simple name.
            /// </summary>
            /// <param name="simpleName">Assembly simple name (for example, <c>Tesseract</c>).</param>
            /// <returns>The loaded assembly if successful; otherwise, null.</returns>
            private static Assembly? TryLoadAssembly(string simpleName) {
                try {
                    return Assembly.Load(new AssemblyName(simpleName));
                }
                catch {
                    return null;
                }
            }

            /// <summary>
            /// Normalizes extracted text by standardizing line endings to <c>\n</c> and collapsing consecutive
            /// non-newline whitespace into single spaces.
            /// </summary>
            /// <param name="s">Input text to normalize.</param>
            /// <returns>A normalized string (never null).</returns>
            private static string NormalizeText(string? s) {
                if (string.IsNullOrWhiteSpace(s)) {
                    return string.Empty;
                }

                string normalized = s.Replace("\r\n", "\n").Replace('\r', '\n');

                StringBuilder sb = new StringBuilder(normalized.Length);
                bool prevSpace = false;

                foreach (char ch in normalized) {
                    if (char.IsWhiteSpace(ch) && ch != '\n') {
                        if (!prevSpace) {
                            sb.Append(' ');
                        }
                        prevSpace = true;
                        continue;
                    }

                    if (ch == '\n') {
                        sb.Append('\n');
                        prevSpace = false;
                        continue;
                    }

                    sb.Append(ch);
                    prevSpace = false;
                }

                return sb.ToString().Trim();
            }

            /// <summary>
            /// Truncates the specified text to a maximum number of characters, optionally cutting at the last newline
            /// within the limit to avoid splitting lines.
            /// </summary>
            /// <remarks>
            /// If a newline character is found within the last quarter of the allowed character range, the text is
            /// truncated at that newline to avoid splitting lines. Otherwise, the text is truncated at the exact
            /// character limit.
            /// </remarks>
            /// <param name="text">The text to truncate.</param>
            /// <param name="maxChars">
            /// The maximum number of characters allowed. If less than or equal to zero, no truncation is performed.
            /// </param>
            /// <returns>A string containing the original text if within the limit; otherwise a truncated string.</returns>
            private static string TruncateIfNeeded(string text, int maxChars) {
                if (maxChars <= 0 || text.Length <= maxChars) {
                    return text;
                }

                ReadOnlySpan<char> slice = text.AsSpan(0, maxChars);
                int lastNewline = slice.LastIndexOf('\n');
                int cut = lastNewline > (int)(maxChars * 0.75) ? lastNewline : maxChars;
                return text.Substring(0, cut);
            }

            /// <summary>
            /// Computes the SHA-256 hash of the contents of the specified file and returns it as a lowercase hex string.
            /// </summary>
            /// <param name="filePath">Path to the file to hash.</param>
            /// <returns>Lowercase hexadecimal string representing the SHA-256 hash.</returns>
            private static string ComputeSha256Hex(string filePath) {
                using SHA256 sha = SHA256.Create();
                using FileStream fs = File.OpenRead(filePath);

                byte[] buffer = new byte[8192];
                int read;

                while ((read = fs.Read(buffer, 0, buffer.Length)) > 0) {
                    sha.TransformBlock(buffer, 0, read, null, 0);
                }

                sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);

                byte[] hash = sha.Hash ?? Array.Empty<byte>();
                return Convert.ToHexString(hash).ToLowerInvariant();
            }

            /// <summary>
            /// Safely returns the first element of the provided array, or <c>0</c> if the array is null or empty.
            /// </summary>
            /// <param name="a">Array to read.</param>
            /// <returns>The first value when available; otherwise <c>0</c>.</returns>
            private static float SafeFirst(float[]? a) {
                if (a == null || a.Length == 0) {
                    return 0f;
                }
                return a[0];
            }

            /// <summary>
            /// Resolves the base artifacts directory for a run by probing common option property names and falling back
            /// to the workflow's configured output directory.
            /// </summary>
            /// <param name="options">Run options that may contain directory fields.</param>
            /// <returns>An absolute directory path.</returns>
            private string ResolveArtifactsDirectory(IWorkflow.WorkflowRunOptions options) {
                string? fromOptions =
                    ReadIfExists<string>(options, "ArtifactsDirectory", null) ??
                    ReadIfExists<string>(options, "OutputDirectory", null) ??
                    ReadIfExists<string>(options, "ArtifactDirectory", null);

                if (!string.IsNullOrWhiteSpace(fromOptions)) {
                    return Path.GetFullPath(fromOptions);
                }

                try {
                    FieldInfo? f = this.GetType().GetField("_options", BindingFlags.Instance | BindingFlags.NonPublic);
                    if (f != null) {
                        object? opt = f.GetValue(this);
                        if (opt != null) {
                            string? dir =
                                ReadIfExists<string>(opt, "OutputDirectory", null) ??
                                ReadIfExists<string>(opt, "ArtifactsDirectory", null);

                            if (!string.IsNullOrWhiteSpace(dir)) {
                                return Path.GetFullPath(dir);
                            }
                        }
                    }
                }
                catch {
                    // ignore
                }

                return Path.Combine(AppContext.BaseDirectory, "artifacts");
            }

            /// <summary>
            /// Best-effort sets the workflow options output directory properties so that training writes into the
            /// per-iteration artifacts folder.
            /// </summary>
            /// <param name="iterationDir">Iteration artifacts directory.</param>
            private void TrySetOptionsOutputDirectory(string iterationDir) {
                try {
                    FieldInfo? f = this.GetType().GetField("_options", BindingFlags.Instance | BindingFlags.NonPublic);
                    if (f == null) {
                        return;
                    }

                    object? opt = f.GetValue(this);
                    if (opt == null) {
                        return;
                    }

                    bool any = false;
                    any |= SetIfExists(opt, "OutputDirectory", iterationDir);
                    any |= SetIfExists(opt, "ArtifactsDirectory", iterationDir);
                    any |= SetIfExists(opt, "ArtifactDirectory", iterationDir);

                    if (any) {
                        _log.LogInformation($"[{Name}] Redirected options artifact output to: {iterationDir}");
                    }
                }
                catch (Exception ex) {
                    _log.LogWarning($"[{Name}] Could not set options output directory: {ex.GetType().Name}: {ex.Message}");
                }
            }

            /// <summary>
            /// Finds the newest file (by <see cref="FileInfo.LastWriteTimeUtc"/>) matching a search pattern within a directory.
            /// </summary>
            /// <param name="directory">Directory to search.</param>
            /// <param name="searchPattern">Search pattern (for example, <c>*.zip</c> or <c>*manifest*.json</c>).</param>
            /// <returns>Absolute file path of the newest match, or null if none are found.</returns>
            private static string? FindNewestFile(string directory, string searchPattern) {
                if (!Directory.Exists(directory)) {
                    return null;
                }

                FileInfo[] files;
                try {
                    DirectoryInfo di = new DirectoryInfo(directory);
                    files = di.GetFiles(searchPattern, SearchOption.AllDirectories);
                }
                catch {
                    return null;
                }

                if (files.Length == 0) {
                    return null;
                }

                FileInfo newest = files
                    .OrderByDescending(f => f.LastWriteTimeUtc)
                    .First();

                return newest.FullName;
            }

            /// <summary>
            /// Attempts to parse clustering quality metrics from a JSON manifest file using a case-insensitive DFS search.
            /// </summary>
            /// <param name="manifestPath">Path to the manifest JSON file.</param>
            /// <returns>A <see cref="ClusteringQuality"/> instance if metrics are found; otherwise null.</returns>
            private static ClusteringQuality? TryReadClusteringQualityFromManifest(string? manifestPath) {
                if (string.IsNullOrWhiteSpace(manifestPath) || !File.Exists(manifestPath)) {
                    return null;
                }

                string json = File.ReadAllText(manifestPath, Encoding.UTF8);
                using JsonDocument doc = JsonDocument.Parse(json);

                double? dbi = TryFindDouble(doc.RootElement, "daviesBouldinIndex");
                double? avgDist = TryFindDouble(doc.RootElement, "averageDistance");

                if (dbi == null && avgDist == null) {
                    return null;
                }

                return new ClusteringQuality {
                    DaviesBouldinIndex = dbi,
                    AverageDistance = avgDist
                };
            }

            /// <summary>
            /// Searches a JSON tree for a numeric property by name (case-insensitive) and returns its double value.
            /// </summary>
            /// <param name="root">Root JSON element to search.</param>
            /// <param name="name">Property name to find (case-insensitive).</param>
            /// <returns>The found value as a double; otherwise null.</returns>
            private static double? TryFindDouble(JsonElement root, string name) {
                Stack<JsonElement> stack = new Stack<JsonElement>();
                stack.Push(root);

                while (stack.Count > 0) {
                    JsonElement e = stack.Pop();

                    if (e.ValueKind == JsonValueKind.Object) {
                        foreach (JsonProperty p in e.EnumerateObject()) {
                            if (string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase)) {
                                if (p.Value.ValueKind == JsonValueKind.Number && p.Value.TryGetDouble(out double d)) {
                                    return d;
                                }
                            }

                            stack.Push(p.Value);
                        }
                    }
                    else if (e.ValueKind == JsonValueKind.Array) {
                        foreach (JsonElement child in e.EnumerateArray()) {
                            stack.Push(child);
                        }
                    }
                }

                return null;
            }

            /// <summary>
            /// Creates an instance of <typeparamref name="T"/> using a parameterless constructor when available.
            /// </summary>
            /// <remarks>
            /// This helper exists to accommodate external workflow contracts that may provide concrete types at runtime.
            /// </remarks>
            /// <typeparam name="T">Type to instantiate.</typeparam>
            /// <returns>An instance of <typeparamref name="T"/>.</returns>
            /// <exception cref="InvalidOperationException">Thrown when the type cannot be instantiated.</exception>
            private static T CreateObject<T>() {
                Type t = typeof(T);

                object? instance = Activator.CreateInstance(t);
                if (instance != null) {
                    return (T)instance;
                }

                throw new InvalidOperationException($"Unable to create an instance of type '{t.FullName}'. Ensure it has a public parameterless constructor.");
            }

            /// <summary>
            /// Sets a public instance property by name if it exists and is writable.
            /// </summary>
            /// <param name="target">Target object to update.</param>
            /// <param name="propertyName">Property name to set.</param>
            /// <param name="value">Value to assign.</param>
            /// <returns><c>true</c> if the property existed and was set; otherwise <c>false</c>.</returns>
            private static bool SetIfExists(object target, string propertyName, object? value) {
                if (target == null) {
                    return false;
                }

                Type t = target.GetType();
                PropertyInfo? p = t.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance);
                if (p == null || !p.CanWrite) {
                    return false;
                }

                try {
                    if (value == null) {
                        p.SetValue(target, null);
                        return true;
                    }

                    Type dest = Nullable.GetUnderlyingType(p.PropertyType) ?? p.PropertyType;
                    Type src = value.GetType();

                    if (dest.IsAssignableFrom(src)) {
                        p.SetValue(target, value);
                        return true;
                    }

                    object converted = Convert.ChangeType(value, dest, CultureInfo.InvariantCulture);
                    p.SetValue(target, converted);
                    return true;
                }
                catch {
                    return false;
                }
            }

            /// <summary>
            /// Reads a public instance property by name if it exists and is readable.
            /// </summary>
            /// <typeparam name="T">Desired return type.</typeparam>
            /// <param name="source">Source object to read from.</param>
            /// <param name="propertyName">Property name to read.</param>
            /// <param name="fallback">Fallback value when missing/unreadable/unconvertible.</param>
            /// <returns>The value when available; otherwise <paramref name="fallback"/>.</returns>
            private static T? ReadIfExists<T>(object source, string propertyName, T? fallback) {
                if (source == null) {
                    return fallback;
                }

                Type t = source.GetType();
                PropertyInfo? p = t.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance);
                if (p == null || !p.CanRead) {
                    return fallback;
                }

                try {
                    object? value = p.GetValue(source);
                    if (value == null) {
                        return fallback;
                    }

                    if (value is T typed) {
                        return typed;
                    }

                    Type dest = typeof(T);
                    object converted = Convert.ChangeType(value, dest, CultureInfo.InvariantCulture);
                    return (T)converted;
                }
                catch {
                    return fallback;
                }
            }

            /// <summary>
            /// Represents a comparable set of clustering quality metrics that can be used to decide whether a new model
            /// should replace a baseline model.
            /// </summary>
            /// <remarks>
            /// Lower values are considered better for both metrics. When both candidates provide a Davies–Bouldin Index,
            /// it is preferred for comparison; otherwise, average distance is used. If the metrics cannot be compared,
            /// the candidate is treated as not better.
            /// </remarks>
            private sealed class ClusteringQuality {
                /// <summary>
                /// Gets or sets the Davies–Bouldin index, where lower values generally indicate better clustering.
                /// </summary>
                public double? DaviesBouldinIndex { get; set; }

                /// <summary>
                /// Gets or sets the average distance metric for clustering (interpretation depends on how it was computed).
                /// Lower is treated as better.
                /// </summary>
                public double? AverageDistance { get; set; }

                /// <summary>
                /// Determines whether this candidate quality is better than the specified baseline.
                /// </summary>
                /// <param name="baseline">Baseline quality to compare against.</param>
                /// <returns><c>true</c> if this instance is considered better than <paramref name="baseline"/>; otherwise <c>false</c>.</returns>
                public bool IsBetterThan(ClusteringQuality baseline) {
                    if (DaviesBouldinIndex.HasValue && baseline.DaviesBouldinIndex.HasValue) {
                        return DaviesBouldinIndex.Value < baseline.DaviesBouldinIndex.Value;
                    }

                    if (AverageDistance.HasValue && baseline.AverageDistance.HasValue) {
                        return AverageDistance.Value < baseline.AverageDistance.Value;
                    }

                    return false;
                }
            }
        }

        /// <summary>
        /// Represents configuration options for training a PDF OCR clustering model.
        /// </summary>
        /// <remarks>
        /// Use this class to specify input PDF files, output locations, clustering parameters, and OCR engine settings.
        /// </remarks>
        public sealed class PdfOcrTrainingOptions {
            /// <summary>
            /// Gets or sets a human-friendly workflow name. If null, a default workflow name is used.
            /// </summary>
            public string? WorkflowName { get; set; }

            /// <summary>
            /// Gets the list of PDF file paths to ingest.
            /// </summary>
            public List<string> PdfFiles { get; set; } = new List<string>();

            /// <summary>
            /// Gets or sets the directory where the trained model and manifest will be written.
            /// </summary>
            public string OutputDirectory { get; set; } = Path.Combine(AppContext.BaseDirectory, "artifacts");

            /// <summary>
            /// Gets or sets the model file name (zip) written to <see cref="OutputDirectory"/>.
            /// </summary>
            public string? ModelFileName { get; set; } = "pdf-ocr-clustering-model.zip";

            /// <summary>
            /// Gets or sets the manifest file name (json) written to <see cref="OutputDirectory"/>.
            /// </summary>
            public string? ManifestFileName { get; set; } = "pdf-ocr-training-manifest.json";

            /// <summary>
            /// Gets or sets the ML.NET random seed used for reproducible training.
            /// </summary>
            public int Seed { get; set; } = 1337;

            /// <summary>
            /// Gets or sets the number of clusters to train for KMeans.
            /// </summary>
            public int NumberOfClusters { get; set; } = 8;

            /// <summary>
            /// Gets or sets the number of scored documents to log as a preview.
            /// </summary>
            public int PreviewCount { get; set; } = 10;

            /// <summary>
            /// Gets or sets the n-gram length used during text featurization.
            /// </summary>
            public int NgramLength { get; set; } = 2;

            /// <summary>
            /// Gets or sets the maximum pages to OCR per PDF (0 or less = no limit).
            /// </summary>
            public int MaxPagesPerPdf { get; set; } = 10;

            /// <summary>
            /// Gets or sets the DPI used for PDF rasterization before OCR.
            /// </summary>
            public int RenderDpi { get; set; } = 200;

            /// <summary>
            /// Gets or sets the minimum extracted character count to treat text as “useful”.
            /// </summary>
            public int MinimumUsefulTextChars { get; set; } = 200;

            /// <summary>
            /// Gets or sets the maximum characters allowed per document after extraction (0 or less = no truncation).
            /// </summary>
            public int MaxCharsPerDocument { get; set; } = 200_000;

            /// <summary>
            /// Gets or sets the Tesseract language code (for example, <c>eng</c> or <c>eng+fra</c>).
            /// </summary>
            public string TesseractLanguage { get; set; } = "eng";

            /// <summary>
            /// Gets or sets the directory containing <c>.traineddata</c> files. If null/empty, defaults to
            /// <c>./tessdata</c> under the application base directory.
            /// </summary>
            public string? TesseractDataPath { get; set; }
        }

        /// <summary>
        /// Manifest describing the output of a PDF OCR training run, including options, artifact paths, and ingest statistics.
        /// </summary>
        public sealed class TrainingManifest {
            /// <summary>
            /// Gets or sets the workflow name that produced this manifest.
            /// </summary>
            public string? WorkflowName { get; set; }

            /// <summary>
            /// Gets or sets the UTC timestamp when this manifest was created.
            /// </summary>
            public DateTimeOffset CreatedUtc { get; set; }

            /// <summary>
            /// Gets or sets the number of documents included in training.
            /// </summary>
            public int DocumentCount { get; set; }

            /// <summary>
            /// Gets or sets the options used for this training run.
            /// </summary>
            public PdfOcrTrainingOptions? Options { get; set; }

            /// <summary>
            /// Gets or sets the on-disk path to the saved model artifact.
            /// </summary>
            public string? ModelPath { get; set; }

            /// <summary>
            /// Gets or sets a small preview set of scored documents and their predicted cluster labels.
            /// </summary>
            public List<ClusterPredictionRow>? Preview { get; set; }

            /// <summary>
            /// Gets or sets per-document ingestion statistics captured during extraction.
            /// </summary>
            public List<DocumentIngestStats>? IngestStats { get; set; }
        }

        /// <summary>
        /// Per-document ingestion statistics captured during PDF text extraction.
        /// </summary>
        public sealed class DocumentIngestStats {
            /// <summary>
            /// Gets or sets the absolute file path of the ingested PDF.
            /// </summary>
            public string? FilePath { get; set; }

            /// <summary>
            /// Gets or sets the SHA-256 hash (hex) of the PDF contents.
            /// </summary>
            public string? Hash { get; set; }

            /// <summary>
            /// Gets or sets whether embedded PDF text was used for this document.
            /// </summary>
            public bool UsedEmbeddedText { get; set; }

            /// <summary>
            /// Gets or sets whether OCR was used for this document.
            /// </summary>
            public bool UsedOcr { get; set; }

            /// <summary>
            /// Gets or sets the number of pages OCR processed for this document.
            /// </summary>
            public int OcrPages { get; set; }

            /// <summary>
            /// Gets or sets the number of extracted characters used for training for this document.
            /// </summary>
            public int ExtractedChars { get; set; }

            /// <summary>
            /// Gets or sets the ingestion duration in milliseconds.
            /// </summary>
            public long DurationMs { get; set; }
        }

        /// <summary>
        /// Training input row representing a document's extracted text content.
        /// </summary>
        public sealed class DocumentTextRow {
            /// <summary>
            /// Gets or sets a stable identifier for the document (typically a SHA-256 hash).
            /// </summary>
            public string DocumentId { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the file name of the input PDF.
            /// </summary>
            public string FileName { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the extracted text used for training and scoring.
            /// </summary>
            [LoadColumn(0)]
            public string Text { get; set; } = string.Empty;
        }

        /// <summary>
        /// Scoring output row for clustering predictions over ingested documents.
        /// </summary>
        public sealed class ClusterPredictionRow {
            /// <summary>
            /// Gets or sets the document identifier associated with this prediction.
            /// </summary>
            public string DocumentId { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the file name of the input document associated with this prediction.
            /// </summary>
            public string FileName { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the predicted cluster label (1-based in many ML.NET clustering trainers).
            /// </summary>
            [ColumnName("PredictedLabel")]
            public uint PredictedLabel { get; set; }

            /// <summary>
            /// Gets or sets the per-cluster score vector produced by the clustering model.
            /// </summary>
            public float[] Score { get; set; } = Array.Empty<float>();
        }

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
            /// <returns>A rendered bitmap that must be disposed by the caller.</returns>
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

        /// <summary>
        /// OCR engine implementation that uses the Tesseract library (loaded via reflection) to recognize text.
        /// </summary>
        /// <remarks>
        /// This type is internal because it is an implementation detail and is loaded conditionally based on runtime
        /// dependencies. Consumers should depend on <see cref="IOcrEngine"/> instead.
        /// </remarks>
        internal sealed class TesseractOcrEngine : IOcrEngine {
            private readonly PdfOcrTrainingOptions _options;

            private readonly object _engine;
            private readonly Type _engineType;
            private readonly MethodInfo _processMethod;
            private readonly MethodInfo _disposeEngineMethod;

            private readonly MethodInfo _getTextMethod;
            private readonly MethodInfo _disposePageMethod;

            public TesseractOcrEngine(PdfOcrTrainingOptions options) {
                _options = options ?? throw new ArgumentNullException(nameof(options));

                Type? engineType = Type.GetType("Tesseract.TesseractEngine, Tesseract", throwOnError: false);
                if (engineType == null) {
                    throw new InvalidOperationException("Tesseract assembly present but TesseractEngine type not resolvable.");
                }
                _engineType = engineType;

                Type? engineModeType = Type.GetType("Tesseract.EngineMode, Tesseract", throwOnError: false);
                if (engineModeType == null) {
                    throw new InvalidOperationException("Tesseract.EngineMode type not resolvable.");
                }

                string dataPath = _options.TesseractDataPath;
                if (string.IsNullOrWhiteSpace(dataPath)) {
                    dataPath = Path.Combine(AppContext.BaseDirectory, "tessdata");
                }

                string lang = _options.TesseractLanguage;
                if (string.IsNullOrWhiteSpace(lang)) {
                    lang = "eng";
                }

                object engineModeDefault = Enum.Parse(engineModeType, "Default", ignoreCase: true);

                object? engine = Activator.CreateInstance(_engineType, dataPath, lang, engineModeDefault);
                if (engine == null) {
                    throw new InvalidOperationException("Failed to create TesseractEngine instance.");
                }
                _engine = engine;

                MethodInfo? process = _engineType.GetMethods()
                    .FirstOrDefault(m => m.Name == "Process" && m.GetParameters().Length == 1);
                if (process == null) {
                    throw new InvalidOperationException("TesseractEngine.Process(...) not found.");
                }
                _processMethod = process;

                MethodInfo? disposeEngine = _engineType.GetMethod("Dispose", Type.EmptyTypes);
                if (disposeEngine == null) {
                    throw new InvalidOperationException("TesseractEngine.Dispose() not found.");
                }
                _disposeEngineMethod = disposeEngine;

                Type? pageType = Type.GetType("Tesseract.Page, Tesseract", throwOnError: false);
                if (pageType == null) {
                    throw new InvalidOperationException("Tesseract.Page type not resolvable.");
                }

                MethodInfo? getText = pageType.GetMethod("GetText", Type.EmptyTypes);
                if (getText == null) {
                    throw new InvalidOperationException("Page.GetText() not found.");
                }
                _getTextMethod = getText;

                MethodInfo? disposePage = pageType.GetMethod("Dispose", Type.EmptyTypes);
                if (disposePage == null) {
                    throw new InvalidOperationException("Page.Dispose() not found.");
                }
                _disposePageMethod = disposePage;
            }

            /// <summary>
            /// Recognizes text from a bitmap using the underlying Tesseract engine.
            /// </summary>
            /// <param name="bitmap">Bitmap to recognize.</param>
            /// <param name="ct">Cancellation token.</param>
            /// <returns>Recognized text (never null).</returns>
            public Task<string> RecognizeAsync(Bitmap bitmap, CancellationToken ct) {
                ct.ThrowIfCancellationRequested();

                return Task.Run(() => {
                    ct.ThrowIfCancellationRequested();

                    object? page = null;
                    try {
                        Type? pixType = Type.GetType("Tesseract.Pix, Tesseract", throwOnError: false);
                        if (pixType == null) {
                            throw new InvalidOperationException("Tesseract.Pix type not resolvable.");
                        }

                        object pix = CreatePixFromBitmap(bitmap, pixType);

                        object? pageObj = _processMethod.Invoke(_engine, new object[] { pix });
                        if (pageObj == null) {
                            throw new InvalidOperationException("Tesseract returned null Page.");
                        }
                        page = pageObj;

                        object? textObj = _getTextMethod.Invoke(page, Array.Empty<object>());
                        string? text = textObj as string;
                        return text ?? string.Empty;
                    }
                    finally {
                        if (page != null) {
                            try {
                                _disposePageMethod.Invoke(page, Array.Empty<object>());
                            }
                            catch {
                                // ignore
                            }
                        }
                    }
                }, ct);
            }

            /// <summary>
            /// Disposes the underlying Tesseract engine instance.
            /// </summary>
            public void Dispose() {
                try {
                    _disposeEngineMethod.Invoke(_engine, Array.Empty<object>());
                }
                catch {
                    // ignore
                }
            }

            /// <summary>
            /// Creates a Tesseract Pix object from a bitmap either via PixConverter or by encoding as PNG and calling Pix.LoadFromMemory.
            /// </summary>
            /// <param name="bitmap">Source bitmap.</param>
            /// <param name="pixType">Pix runtime type.</param>
            /// <returns>A Pix object instance.</returns>
            private static object CreatePixFromBitmap(Bitmap bitmap, Type pixType) {
                Type? pixConverterType = Type.GetType("Tesseract.PixConverter, Tesseract", throwOnError: false);
                if (pixConverterType != null) {
                    MethodInfo? toPix = pixConverterType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                        .FirstOrDefault(m => m.Name == "ToPix" && m.GetParameters().Length == 1);
                    if (toPix != null) {
                        object? pixObj = toPix.Invoke(null, new object[] { bitmap });
                        if (pixObj != null) {
                            return pixObj;
                        }
                    }
                }

                MethodInfo? loadFromMemory = pixType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .FirstOrDefault(m => m.Name == "LoadFromMemory" && m.GetParameters().Length == 1);
                if (loadFromMemory == null) {
                    throw new InvalidOperationException("No Pix conversion available (PixConverter.ToPix or Pix.LoadFromMemory).");
                }

                using MemoryStream ms = new MemoryStream();
                bitmap.Save(ms, ImageFormat.Png);
                byte[] bytes = ms.ToArray();

                object? pix2 = loadFromMemory.Invoke(null, new object[] { bytes });
                if (pix2 == null) {
                    throw new InvalidOperationException("Pix.LoadFromMemory returned null.");
                }

                return pix2;
            }
        }

        /// <summary>
        /// PDF rasterizer implementation using PdfiumViewer (loaded via reflection).
        /// </summary>
        /// <remarks>
        /// This type is internal because it is an implementation detail and is loaded conditionally based on runtime
        /// dependencies. Consumers should depend on <see cref="IPdfRasterizer"/> instead.
        /// </remarks>
        internal sealed class PdfiumViewerRasterizer : IPdfRasterizer {
            private readonly Type _pdfDocumentType;
            private readonly MethodInfo _loadMethod;
            private readonly PropertyInfo _pageCountProp;
            private readonly MethodInfo _disposeDocMethod;

            public PdfiumViewerRasterizer() {
                Type? docType = Type.GetType("PdfiumViewer.PdfDocument, PdfiumViewer", throwOnError: false);
                if (docType == null) {
                    throw new InvalidOperationException("PdfiumViewer.PdfDocument type not resolvable.");
                }
                _pdfDocumentType = docType;

                MethodInfo? load = _pdfDocumentType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .FirstOrDefault(m => m.Name == "Load" &&
                                         m.GetParameters().Length == 1 &&
                                         m.GetParameters()[0].ParameterType == typeof(string));
                if (load == null) {
                    throw new InvalidOperationException("PdfDocument.Load(string) not found.");
                }
                _loadMethod = load;

                PropertyInfo? pageCount = _pdfDocumentType.GetProperty("PageCount", BindingFlags.Public | BindingFlags.Instance);
                if (pageCount == null) {
                    throw new InvalidOperationException("PdfDocument.PageCount not found.");
                }
                _pageCountProp = pageCount;

                MethodInfo? dispose = _pdfDocumentType.GetMethod("Dispose", Type.EmptyTypes);
                if (dispose == null) {
                    throw new InvalidOperationException("PdfDocument.Dispose() not found.");
                }
                _disposeDocMethod = dispose;
            }

            /// <summary>
            /// Gets the page count for a given PDF file.
            /// </summary>
            /// <param name="pdfPath">Path to a PDF file.</param>
            /// <returns>Page count reported by PdfiumViewer.</returns>
            public int GetPageCount(string pdfPath) {
                object doc = Load(pdfPath);
                try {
                    object? value = _pageCountProp.GetValue(doc);
                    return value is int i ? i : 0;
                }
                finally {
                    _disposeDocMethod.Invoke(doc, Array.Empty<object>());
                }
            }

            /// <summary>
            /// Renders a single page of a PDF into a bitmap using PdfiumViewer.
            /// </summary>
            /// <param name="pdfPath">Path to a PDF file.</param>
            /// <param name="pageIndex">Zero-based page index.</param>
            /// <param name="dpi">Target DPI for rendering.</param>
            /// <returns>A bitmap that must be disposed by the caller.</returns>
            public Bitmap RenderPage(string pdfPath, int pageIndex, int dpi) {
                object doc = Load(pdfPath);
                try {
                    MethodInfo? render4 = _pdfDocumentType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                        .FirstOrDefault(m => {
                            if (m.Name != "Render")
                                return false;
                            ParameterInfo[] p = m.GetParameters();
                            if (p.Length != 4)
                                return false;
                            return p[0].ParameterType == typeof(int) &&
                                   p[1].ParameterType == typeof(int) &&
                                   p[2].ParameterType == typeof(int);
                        });

                    Type? flagsType = Type.GetType("PdfiumViewer.PdfRenderFlags, PdfiumViewer", throwOnError: false);
                    object flags = flagsType == null ? 0 : Enum.Parse(flagsType, "ForPrinting", ignoreCase: true);

                    if (render4 != null) {
                        object? bmpObj = render4.Invoke(doc, new object[] { pageIndex, dpi, dpi, flags });
                        Bitmap? bmp = bmpObj as Bitmap;
                        if (bmp != null) {
                            return bmp;
                        }
                    }

                    MethodInfo? render6 = _pdfDocumentType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                        .FirstOrDefault(m => {
                            if (m.Name != "Render")
                                return false;
                            ParameterInfo[] p = m.GetParameters();
                            if (p.Length != 6)
                                return false;
                            return p[0].ParameterType == typeof(int) &&
                                   p[1].ParameterType == typeof(int) &&
                                   p[2].ParameterType == typeof(int) &&
                                   p[3].ParameterType == typeof(int) &&
                                   p[4].ParameterType == typeof(int);
                        });

                    if (render6 == null) {
                        throw new InvalidOperationException("PdfiumViewer.Render overloads not found.");
                    }

                    int width = (int)(8.27 * dpi);
                    int height = (int)(11.69 * dpi);

                    object? bmpObj2 = render6.Invoke(doc, new object[] { pageIndex, width, height, dpi, dpi, flags });
                    Bitmap? bmp2 = bmpObj2 as Bitmap;
                    if (bmp2 == null) {
                        throw new InvalidOperationException("PdfiumViewer.Render returned null Bitmap.");
                    }
                    return bmp2;
                }
                finally {
                    _disposeDocMethod.Invoke(doc, Array.Empty<object>());
                }
            }

            /// <summary>
            /// Disposes the rasterizer instance (no-op for this implementation).
            /// </summary>
            public void Dispose() {
                // no-op
            }

            /// <summary>
            /// Loads a PDF document using PdfiumViewer's <c>PdfDocument.Load</c>.
            /// </summary>
            /// <param name="pdfPath">Path to the PDF document.</param>
            /// <returns>A PdfiumViewer PdfDocument instance.</returns>
            private object Load(string pdfPath) {
                object? doc = _loadMethod.Invoke(null, new object[] { pdfPath });
                if (doc == null) {
                    throw new InvalidOperationException("PdfDocument.Load returned null.");
                }
                return doc;
            }
        }

        /// <summary>
        /// Embedded-text extractor implementation using UglyToad.PdfPig (loaded via reflection).
        /// </summary>
        /// <remarks>
        /// This type is internal because it is an implementation detail and is loaded conditionally based on runtime
        /// dependencies. Consumers should depend on <see cref="IPdfTextExtractor"/> instead.
        /// </remarks>
        internal sealed class PdfPigTextExtractor : IPdfTextExtractor {
            private readonly Type _pdfDocumentType;
            private readonly MethodInfo _openMethod;
            private readonly PropertyInfo _numberOfPagesProp;
            private readonly MethodInfo _getPageMethod;
            private readonly PropertyInfo _pageTextProp;
            private readonly MethodInfo _disposeDocMethod;
            private readonly MethodInfo _disposePageMethod;

            public PdfPigTextExtractor() {
                Type? docType = Type.GetType("UglyToad.PdfPig.PdfDocument, UglyToad.PdfPig", throwOnError: false);
                if (docType == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument type not resolvable.");
                }
                _pdfDocumentType = docType;

                MethodInfo? open = _pdfDocumentType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .FirstOrDefault(m => m.Name == "Open" &&
                                         m.GetParameters().Length == 1 &&
                                         m.GetParameters()[0].ParameterType == typeof(string));
                if (open == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument.Open(string) not found.");
                }
                _openMethod = open;

                PropertyInfo? numberOfPages = _pdfDocumentType.GetProperty("NumberOfPages", BindingFlags.Public | BindingFlags.Instance);
                if (numberOfPages == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument.NumberOfPages not found.");
                }
                _numberOfPagesProp = numberOfPages;

                MethodInfo? getPage = _pdfDocumentType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .FirstOrDefault(m => m.Name == "GetPage" &&
                                         m.GetParameters().Length == 1 &&
                                         m.GetParameters()[0].ParameterType == typeof(int));
                if (getPage == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument.GetPage(int) not found.");
                }
                _getPageMethod = getPage;

                Type? pageType = Type.GetType("UglyToad.PdfPig.Content.Page, UglyToad.PdfPig", throwOnError: false);
                if (pageType == null) {
                    throw new InvalidOperationException("PdfPig Page type not resolvable.");
                }

                PropertyInfo? textProp = pageType.GetProperty("Text", BindingFlags.Public | BindingFlags.Instance);
                if (textProp == null) {
                    throw new InvalidOperationException("PdfPig Page.Text not found.");
                }
                _pageTextProp = textProp;

                MethodInfo? disposeDoc = _pdfDocumentType.GetMethod("Dispose", Type.EmptyTypes);
                if (disposeDoc == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument.Dispose() not found.");
                }
                _disposeDocMethod = disposeDoc;

                MethodInfo? disposePage = pageType.GetMethod("Dispose", Type.EmptyTypes);
                if (disposePage == null) {
                    throw new InvalidOperationException("PdfPig Page.Dispose() not found.");
                }
                _disposePageMethod = disposePage;
            }

            /// <summary>
            /// Extracts embedded text from a PDF file using PdfPig.
            /// </summary>
            /// <param name="pdfPath">Path to a PDF file.</param>
            /// <returns>Concatenated text content from all pages.</returns>
            public string ExtractAllText(string pdfPath) {
                object? docObj = _openMethod.Invoke(null, new object[] { pdfPath });
                if (docObj == null) {
                    throw new InvalidOperationException("PdfPig PdfDocument.Open returned null.");
                }

                try {
                    object? pagesObj = _numberOfPagesProp.GetValue(docObj);
                    int pages = pagesObj is int i ? i : 0;

                    StringBuilder sb = new StringBuilder(16_384);

                    for (int pageNumber = 1; pageNumber <= pages; pageNumber++) {
                        object? pageObj = _getPageMethod.Invoke(docObj, new object[] { pageNumber });
                        if (pageObj == null) {
                            continue;
                        }

                        try {
                            object? textObj = _pageTextProp.GetValue(pageObj);
                            string? text = textObj as string;
                            if (!string.IsNullOrWhiteSpace(text)) {
                                sb.AppendLine(text);
                                sb.AppendLine();
                            }
                        }
                        finally {
                            try {
                                _disposePageMethod.Invoke(pageObj, Array.Empty<object>());
                            }
                            catch {
                                // ignore
                            }
                        }
                    }

                    return sb.ToString();
                }
                finally {
                    try {
                        _disposeDocMethod.Invoke(docObj, Array.Empty<object>());
                    }
                    catch {
                        // ignore
                    }
                }
            }

            /// <summary>
            /// Disposes the extractor instance (no-op for this implementation).
            /// </summary>
            public void Dispose() {
                // no-op
            }
        }

        /// <summary>
        /// Common JSON serializer options used for persisted workflow artifacts.
        /// </summary>
        internal static class JsonOptions {
            /// <summary>
            /// Gets preconfigured options suitable for writing readable manifests.
            /// </summary>
            public static JsonSerializerOptions Pretty { get; } = new JsonSerializerOptions {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
        }
    }
}
