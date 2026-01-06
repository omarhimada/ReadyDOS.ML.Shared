using System.Text.Json;

namespace ReadyDOS.ML.Shared.Utility.FradulentImageDetection {

    /// <summary>
    /// Provides static methods for encoding and decoding run-length encoded (RLE) masks, calculating F1 scores between
    /// binary masks, and evaluating segmentation predictions using optimal assignment metrics compatible with
    /// Python-based RLE conventions.
    /// </summary>
    /// <remarks>This class is designed for interoperability with Python RLE formats, using Fortran
    /// (column-major) order for mask flattening and JSON array representations for RLE data. It includes utilities for
    /// scoring segmentation predictions, including optimal F1 assignment using the Hungarian algorithm, and supports
    /// evaluation workflows similar to those used in data science competitions. All methods are thread-safe as they do
    /// not maintain internal state.</remarks>
    public static class RunLengthEncodingF1Scoring {
        public sealed class ParticipantVisibleError : Exception {
            public ParticipantVisibleError(string message) : base(message) { }
            public ParticipantVisibleError(string message, Exception inner) : base(message, inner) { }
        }


        /// <summary>
        /// Encodes a collection of binary masks using run-length encoding (RLE) and returns the result as a
        /// semicolon-separated string of JSON-serialized RLE data.
        /// </summary>
        /// <remarks>The output string can be parsed by splitting on semicolons and deserializing each
        /// segment from JSON to obtain the RLE data for each mask. This method is useful for compactly representing
        /// multiple binary masks for storage or transmission.</remarks>
        /// <param name="masks">A collection of two-dimensional boolean arrays representing the binary masks to encode. Each mask is
        /// processed independently.</param>
        /// <param name="fgVal">The value in the mask arrays that is considered foreground and will be encoded. Defaults to <see
        /// langword="true"/>.</param>
        /// <returns>A semicolon-separated string where each segment is a JSON-serialized RLE representation of the corresponding
        /// mask in <paramref name="masks"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="masks"/> is <see langword="null"/>.</exception>
        public static string RleEncode(IReadOnlyList<bool[,]> masks, bool fgVal = true) {
            ArgumentNullException.ThrowIfNull(masks);
            return string.Join(";", masks.Select(m => JsonSerializer.Serialize(RleEncodeSingle(m, fgVal))));
        }

        /// <summary>
        /// Decodes a run-length encoded (RLE) mask from a JSON array into a two-dimensional boolean array representing
        /// the mask.
        /// </summary>
        /// <remarks>The input JSON array must contain pairs of integers, where each pair specifies the
        /// start index and length of a run of masked pixels. Start positions must be in non-decreasing order. This
        /// method is typically used to efficiently store and reconstruct binary masks for image processing
        /// tasks.</remarks>
        /// <param name="maskRleJson">A JSON-formatted string containing an array of integers representing the RLE-encoded mask. The array must
        /// consist of pairs of start positions and run lengths, in ascending order.</param>
        /// <param name="height">The number of rows in the resulting mask. Must be a positive integer.</param>
        /// <param name="width">The number of columns in the resulting mask. Must be a positive integer.</param>
        /// <returns>A two-dimensional boolean array of size <paramref name="height"/> by <paramref name="width"/>, where <see
        /// langword="true"/> indicates a masked pixel and <see langword="false"/> indicates an unmasked pixel.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="maskRleJson"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="height"/> or <paramref name="width"/> is less than or equal to zero.</exception>
        /// <exception cref="ParticipantVisibleError">Thrown if <paramref name="maskRleJson"/> is not a valid JSON array of integers, if the start positions are
        /// not in ascending order, or if the RLE data is otherwise invalid.</exception>
        public static bool[,] RleDecode(string maskRleJson, int height, int width) {
            ArgumentNullException.ThrowIfNull(maskRleJson);
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height));
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width));

            int[] pairs;
            try {
                pairs = JsonSerializer.Deserialize<int[]>(maskRleJson) ?? Array.Empty<int>();
            } catch (Exception e) {
                throw new ParticipantVisibleError("RLE must be valid JSON array of ints.", e);
            }

            // Ascending start validation (Python does this before decode)
            List<int> starts = [];
            for (int i = 0; i < pairs.Length; i += 2) {
                if (i >= pairs.Length)
                    break;
                starts.Add(pairs[i]);
            }
            if (!IsNonDecreasing(starts))
                throw new ParticipantVisibleError("Submitted values must be in ascending order.");

            try {
                return RleDecodePairsChecked(pairs, height, width);
            } catch (ArgumentException e) {
                throw new ParticipantVisibleError(e.Message, e);
            }
        }

        /// <summary>
        /// Calculates the optimal F1 score between predicted and ground truth segmentation masks for a single image.
        /// </summary>
        /// <remarks>This method is typically used for evaluating instance segmentation results on a
        /// per-image basis. The F1 score is computed by optimally matching predicted and ground truth masks based on
        /// their overlap.</remarks>
        /// <param name="labelRles">A semicolon-separated string of run-length encoded (RLE) ground truth masks for the image. Each RLE
        /// represents a single object mask.</param>
        /// <param name="predictionRles">A semicolon-separated string of run-length encoded (RLE) predicted masks for the image. Each RLE represents
        /// a single predicted object mask.</param>
        /// <param name="shapeJson">A JSON array representing the image shape in the format [height, width]. Must contain exactly two integers.</param>
        /// <returns>The optimal F1 score computed between the predicted and ground truth masks. The value ranges from 0.0 (no
        /// overlap) to 1.0 (perfect match).</returns>
        /// <exception cref="ArgumentNullException">Thrown if either labelRles, predictionRles, or shapeJson is null.</exception>
        /// <exception cref="ParticipantVisibleError">Thrown if shapeJson is not valid JSON, is empty, or does not represent exactly two integers in the format
        /// [height, width].</exception>
        public static double EvaluateSingleImage(string labelRles, string predictionRles, string shapeJson) {
            ArgumentNullException.ThrowIfNull(labelRles);
            ArgumentNullException.ThrowIfNull(predictionRles);
            ArgumentNullException.ThrowIfNull(shapeJson);

            int[] shape;
            try {
                shape = JsonSerializer.Deserialize<int[]>(shapeJson) ?? throw new ParticipantVisibleError("Shape JSON is empty.");
            } catch (Exception e) {
                throw new ParticipantVisibleError("Shape must be valid JSON like [height, width].", e);
            }

            if (shape.Length != 2)
                throw new ParticipantVisibleError("Shape must have exactly 2 integers: [height, width].");

            int height = shape[0];
            int width = shape[1];

            List<bool[,]> labelMasks =
                labelRles
                    .Split(';', StringSplitOptions.RemoveEmptyEntries)
                        .Select(r => RleDecode(r, height, width))
                        .ToList();

            List<bool[,]> predMasks =
                predictionRles
                    .Split(';', StringSplitOptions.RemoveEmptyEntries)
                    .Select(r => RleDecode(r, height, width))
                    .ToList();

            return OptimalF1Score(predMasks, labelMasks);
        }

        /// <summary>
        /// Calculates the F1 score between a predicted binary mask and a ground truth binary mask.
        /// </summary>
        /// <remarks>The F1 score is the harmonic mean of precision and recall, commonly used to evaluate
        /// the accuracy of binary classification tasks such as image segmentation.</remarks>
        /// <param name="predMask">A two-dimensional Boolean array representing the predicted mask. Each element indicates whether the
        /// corresponding pixel is predicted as positive.</param>
        /// <param name="gtMask">A two-dimensional Boolean array representing the ground truth mask. Each element indicates whether the
        /// corresponding pixel is actually positive.</param>
        /// <returns>The F1 score as a value between 0.0 and 1.0, where 1.0 indicates perfect overlap between the predicted and
        /// ground truth masks. Returns 0.0 if there are no true positives, false positives, or false negatives.</returns>
        /// <exception cref="ArgumentNullException">Thrown if either predMask or gtMask is null.</exception>
        /// <exception cref="ArgumentException">Thrown if predMask and gtMask do not have the same dimensions.</exception>
        public static double CalculateF1Score(bool[,] predMask, bool[,] gtMask) {
            ArgumentNullException.ThrowIfNull(predMask);
            ArgumentNullException.ThrowIfNull(gtMask);

            int height = predMask.GetLength(0);
            int width = predMask.GetLength(1);
            if (gtMask.GetLength(0) != height || gtMask.GetLength(1) != width) {
                throw new ArgumentException("predMask and gtMask must have the same shape.");
            }

            long tp = 0, fp = 0, fn = 0;

            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++) {
                    bool predicted = predMask[r, c];
                    bool groundTruth = gtMask[r, c];

                    if (predicted && groundTruth)
                        tp++;
                    else if (predicted && !groundTruth)
                        fp++;
                    else if (!predicted && groundTruth)
                        fn++;
                }
            }

            double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;

            return (precision + recall) > 0
                ? 2.0 * (precision * recall) / (precision + recall)
                : 0.0;
        }

        public static double[,] CalculateF1Matrix(IReadOnlyList<bool[,]> predMasks, IReadOnlyList<bool[,]> gtMasks) {
            ArgumentNullException.ThrowIfNull(predMasks);
            ArgumentNullException.ThrowIfNull(gtMasks);

            int numPred = predMasks.Count;
            int numGt = gtMasks.Count;

            double[,] matrix = new double[numPred, numGt];

            for (int i = 0; i < numPred; i++) {
                for (int j = 0; j < numGt; j++) {
                    matrix[i, j] = CalculateF1Score(predMasks[i], gtMasks[j]);
                }
            }

            // Python pads with extra zero-rows if preds < gts
            if (numPred < numGt) {
                double[,] padded = new double[numGt, numGt];
                for (int i = 0; i < numPred; i++)
                    for (int j = 0; j < numGt; j++)
                        padded[i, j] = matrix[i, j];
                // remaining rows are already zeros
                return padded;
            }

            return matrix;
        }

        /// <summary>
        /// Optimal F1 using Hungarian assignment on -F1 matrix, with the same penalty term as Python:
        /// excess_predictions_penalty = len(gt) / max(len(pred), len(gt))
        /// return mean(assigned f1) * penalty
        /// </summary>
        public static double OptimalF1Score(IReadOnlyList<bool[,]> predMasks, IReadOnlyList<bool[,]> gtMasks) {
            ArgumentNullException.ThrowIfNull(predMasks);
            ArgumentNullException.ThrowIfNull(gtMasks);

            if (gtMasks.Count == 0 && predMasks.Count == 0)
                return 1.0;
            if (gtMasks.Count == 0)
                return 0.0; // assignment mean 0 * penalty
            if (predMasks.Count == 0)
                return 0.0;

            double[,] f1 = CalculateF1Matrix(predMasks, gtMasks);
            int rows = f1.GetLength(0);
            int cols = f1.GetLength(1);

            // Convert to a cost matrix for minimization: cost = -f1
            double[,] cost = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    cost[i, j] = -f1[i, j];

            (int[] rowInd, int[] colInd) = HungarianMinimize(cost);

            double sum = 0.0;
            int n = rowInd.Length;
            for (int k = 0; k < n; k++)
                sum += f1[rowInd[k], colInd[k]];

            double mean = n > 0 ? sum / n : 0.0;

            double penalty = (double)gtMasks.Count / Math.Max(predMasks.Count, gtMasks.Count);
            return mean * penalty;
        }

        /// <summary>
        /// Represents a single row in a score table, containing an identifier, annotation, and shape information.
        /// </summary>
        /// <param name="RowId">The unique identifier for the score row. Cannot be null.</param>
        /// <param name="Annotation">The annotation or descriptive text associated with the score row. Cannot be null.</param>
        /// <param name="Shape">The shape or format information for the score row. Cannot be null.</param>
        public sealed record ScoreRow(string RowId, string Annotation, string Shape);

        /// <summary>
        /// Calculates the average score comparing a solution against a submission, based on annotation matches for each
        /// row.
        /// </summary>
        /// <remarks>Each row is compared by its annotation. If either the solution or submission
        /// annotation is 'authentic', the score for that row is 1.0 if the annotations match exactly; otherwise, it is
        /// 0.0. For other cases, a custom evaluation is performed per image. The method does not modify the input
        /// lists.</remarks>
        /// <param name="solution">The list of reference rows representing the correct solution. Cannot be null and must have the same number
        /// of elements as <paramref name="submission"/>.</param>
        /// <param name="submission">The list of rows to be evaluated against the solution. Cannot be null and must have the same number of
        /// elements as <paramref name="solution"/>.</param>
        /// <returns>A double value representing the average score across all rows. The score ranges from 0.0 to 1.0, where 1.0
        /// indicates a perfect match.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="solution"/> or <paramref name="submission"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="solution"/> and <paramref name="submission"/> do not contain the same number of
        /// rows.</exception>
        public static double Score(IReadOnlyList<ScoreRow> solution, IReadOnlyList<ScoreRow> submission) {
            ArgumentNullException.ThrowIfNull(solution);
            ArgumentNullException.ThrowIfNull(submission);
            if (solution.Count != submission.Count)
                throw new ArgumentException("Solution and submission must have the same number of rows.");

            double total = 0.0;

            for (int i = 0; i < solution.Count; i++) {
                var sol = solution[i];
                var sub = submission[i];

                string label = sol.Annotation;
                string pred = sub.Annotation;

                bool authenticIndices = string.Equals(label, "authentic", StringComparison.Ordinal) ||
                                        string.Equals(pred, "authentic", StringComparison.Ordinal);

                double imageScore;

                if (authenticIndices) {
                    imageScore = string.Equals(label, pred, StringComparison.Ordinal) ? 1.0 : 0.0;
                } else {
                    imageScore = EvaluateSingleImage(label, pred, sol.Shape);
                }

                total += imageScore;
            }

            return total / solution.Count;
        }

        /// <summary>
        /// Encodes a 2D boolean mask into run-length encoding (RLE) format for a single foreground value.
        /// </summary>
        /// <remarks>The mask is traversed in column-major (Fortran) order. The returned list alternates
        /// between start positions and run lengths, suitable for use with image processing or segmentation
        /// tasks.</remarks>
        /// <param name="mask">A two-dimensional boolean array representing the mask to encode. Each element indicates whether the
        /// corresponding position is foreground or background.</param>
        /// <param name="fgVal">The boolean value in the mask to treat as the foreground when generating the run-length encoding.</param>
        /// <returns>A list of integers representing the run-length encoding of the specified foreground value in the mask. Each
        /// pair of values corresponds to the start position (1-based, in column-major order) and the length of a run.</returns>
        private static List<int> RleEncodeSingle(bool[,] mask, bool fgVal) {
            ArgumentNullException.ThrowIfNull(mask);
            int h = mask.GetLength(0);
            int w = mask.GetLength(1);

            // index = row + (col * height)
            List<int> dots = new(capacity: h * w / 8);

            for (int c = 0; c < w; c++) {
                for (int r = 0; r < h; r++) {
                    if (mask[r, c] == fgVal) {
                        dots.Add(r + c * h);
                    }
                }
            }

            List<int> runLengths = [];
            int prev = -2;

            foreach (int b in dots) {
                if (b > prev + 1) {
                    runLengths.Add(b + 1); // 1-based start
                    runLengths.Add(0);     // length placeholder
                }
                runLengths[runLengths.Count - 1] += 1;
                prev = b;
            }

            return runLengths;
        }

        private static bool[,] RleDecodePairsChecked(int[] pairs, int height, int width) {
            if (pairs.Length % 2 != 0)
                throw new ArgumentException("One or more rows has an odd number of values.");

            int n = pairs.Length / 2;
            int[] starts = new int[n];
            int[] lengths = new int[n];
            for (int i = 0; i < n; i++) {
                starts[i] = pairs[2 * i];
                lengths[i] = pairs[2 * i + 1];
            }

            // Convert to 0-based starts
            for (int i = 0; i < n; i++) {
                starts[i] -= 1;
            }

            int[] ends = new int[n];
            for (int i = 0; i < n; i++) {
                ends[i] = starts[i] + lengths[i];
            }

            for (int i = 0; i < n - 1; i++) {
                if (ends[i] > starts[i + 1]) {
                    throw new ArgumentException("Pixels must not be overlapping.");
                }
            }

            int total = checked(height * width);
            bool[] flat = new bool[total];
            for (int i = 0; i < n; i++) {
                int lo = starts[i];
                int hi = ends[i];
                if (lo < 0 || hi < 0 || lo > total || hi > total) {
                    throw new ArgumentException("RLE indices are out of bounds for the provided shape.");
                }
                for (int k = lo; k < hi; k++)
                    flat[k] = true;
            }

            // Reshape from flat Fortran-order into [height,width]
            bool[,] mask = new bool[height, width];
            int idx = 0;
            for (int c = 0; c < width; c++) {
                for (int r = 0; r < height; r++) {
                    mask[r, c] = flat[idx++];
                }
            }

            return mask;
        }

        /// <summary>
        /// Determines whether the elements in the specified list are in non-decreasing order.
        /// </summary>
        /// <param name="xs">The list of integers to check for non-decreasing order. Cannot be null.</param>
        /// <returns><see langword="true"/> if the elements in <paramref name="xs"/> are in non-decreasing order; otherwise, <see
        /// langword="false"/>.</returns>
        private static bool IsNonDecreasing(IReadOnlyList<int> xs) {
            for (int i = 1; i < xs.Count; i++) {
                if (xs[i] < xs[i - 1]) {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Finds an assignment of rows to columns that minimizes the total cost using the Hungarian algorithm.
        /// </summary>
        /// <remarks>If the number of rows exceeds the number of columns, the method automatically
        /// transposes the cost matrix to ensure a valid assignment and returns the result in terms of the original
        /// matrix's row and column indices. The method finds an assignment for all rows, assigning each to a unique
        /// column, such that the total cost is minimized. The input matrix is not modified.</remarks>
        /// <param name="cost">A two-dimensional array representing the cost matrix, where each element at position [i, j] specifies the
        /// cost of assigning row i to column j. The array must have at least one row and one column.</param>
        /// <returns>A tuple containing two arrays: the first array contains the indices of the assigned rows, and the second
        /// array contains the corresponding indices of the assigned columns. Each pair (rowInd[i], colInd[i])
        /// represents an assignment included in the minimum-cost matching.</returns>
        private static (int[] rowInd, int[] colInd) HungarianMinimize(double[,] cost) {
            int rows = cost.GetLength(0);
            int cols = cost.GetLength(1);

            // Hungarian is typically described for rows <= cols. If not, transpose.
            bool transposed = false;
            double[,] a = cost;

            if (rows > cols) {
                transposed = true;
                a = Transpose(cost);
                (rows, cols) = (cols, rows);
            }

            // Now rows <= cols
            // Implementation based on classic O(n^3) potentials approach.
            // u: row potentials (1..rows), v: col potentials (1..cols)
            // predicted: matching for cols (0..cols), way: predecessor cols
            double[] u = new double[rows + 1];
            double[] v = new double[cols + 1];
            int[] p = new int[cols + 1];
            int[] way = new int[cols + 1];

            for (int i = 1; i <= rows; i++) {
                p[0] = i;
                int j0 = 0;

                double[] minv = new double[cols + 1];
                bool[] used = new bool[cols + 1];
                for (int j = 1; j <= cols; j++)
                    minv[j] = double.PositiveInfinity;

                do {
                    used[j0] = true;
                    int i0 = p[j0];
                    double delta = double.PositiveInfinity;
                    int j1 = 0;

                    for (int j = 1; j <= cols; j++) {
                        if (used[j])
                            continue;

                        double cur = a[i0 - 1, j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }

                    for (int j = 0; j <= cols; j++) {
                        if (used[j]) {
                            u[p[j]] += delta;
                            v[j] -= delta;
                        } else {
                            minv[j] -= delta;
                        }
                    }

                    j0 = j1;
                }
                while (p[j0] != 0);

                // Augmenting
                do {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                }
                while (j0 != 0);
            }

            // Build assignment: row -> col
            // predicted[j] = matched row for column j
            int[] rowToCol = new int[rows + 1];
            for (int j = 1; j <= cols; j++) {
                int i = p[j];
                if (i != 0)
                    rowToCol[i] = j;
            }

            int[] rowInd = new int[rows];
            int[] colInd = new int[rows];
            for (int i = 1; i <= rows; i++) {
                rowInd[i - 1] = i - 1;
                colInd[i - 1] = rowToCol[i] - 1;
            }

            if (!transposed) {
                return (rowInd, colInd);
            }

            // If transposed, swap back
            int[] origRowInd = colInd;
            int[] origColInd = rowInd;

            return (origRowInd, origColInd);
        }

        /// <summary>
        /// Returns a new matrix that is the transpose of the specified two-dimensional array.
        /// </summary>
        /// <param name="m">The two-dimensional array of double values to transpose. Must not be null.</param>
        /// <returns>A new two-dimensional array containing the transposed values of the input matrix.</returns>
        private static double[,] Transpose(double[,] m) {
            int r = m.GetLength(0);
            int c = m.GetLength(1);
            double[,] t = new double[c, r];
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    t[j, i] = m[i, j];
                }
            }
            return t;
        }
    }

}
