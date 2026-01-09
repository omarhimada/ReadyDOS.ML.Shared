using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.Recommender;
using System.Data;

namespace ReadyDOS.ML.Shared.PrecisionAtK {
    public static class Compute {
        /// <summary>
        /// Calculates the average precision at K for a matrix factorization recommendation model over a set of
        /// evaluation data.
        /// </summary>
        /// <remarks>Precision at K measures the proportion of recommended items in the top K that are
        /// relevant (i.e., actually purchased) for each user, averaged over all users. This metric is commonly used to
        /// evaluate the quality of recommendation systems. The method assumes that the evaluation data contains at
        /// least one interaction per user and that all SKUs to be recommended are present in the evaluation
        /// data.</remarks>
        /// <typeparam name="T">The type of the evaluation data, which must inherit from Order and represent a user-item interaction.</typeparam>
        /// <param name="mlContext">The ML.NET context to use for data operations and transformations.</param>
        /// <param name="model">The trained matrix factorization prediction transformer used to generate recommendation scores.</param>
        /// <param name="evalData">The collection of evaluation data representing actual user-item interactions. Each item should correspond to
        /// a purchase or positive interaction.</param>
        /// <param name="encodingModel">The data transformation pipeline used to encode candidate data into the format expected by the model.</param>
        /// <param name="k">The number of top recommendations to consider for each user. Must be a positive integer. Defaults to 5.</param>
        /// <returns>A double value representing the mean precision at K across all users in the evaluation data. Returns 0.0 if
        /// the evaluation data is empty.</returns>
        public static double PrecisionAtK<T> (
            MLContext mlContext,
            MatrixFactorizationPredictionTransformer model,
            IEnumerable<T> evalData,
            ITransformer encodingModel,
            int k = 5) where T : Order {

            ArgumentNullException.ThrowIfNull(mlContext);
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(evalData);
            ArgumentNullException.ThrowIfNull(encodingModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

            // Materialize once so we don't re-enumerate evalData repeatedly
            IList<T> evalList = evalData as IList<T> ?? evalData.ToList();
            if (evalList.Count == 0) {
                return 0.0;
            }

            // Group actual purchases by customer
            // NOTE: your original code had a small bug in ToDictionary/ToHashSet placement; fixed here.
            Dictionary<int, HashSet<int>> actualByCustomer = evalList
                .GroupBy(c => c.CustomerId)
                .ToDictionary(
                    g => g.Key,
                    g => g.Select(x => x.Sku).ToHashSet()
                );

            // Universe of SKUs to consider for recommendations
            // (same behavior as your original method)
            int[] allSkus = evalList.Select(c => c.Sku).Distinct().ToArray();

            double totalPrecision = 0.0;
            int userCount = 0;

            foreach (int customerId in actualByCustomer.Keys) {
                // Build candidate rows: (customerId, sku) for every sku
                // This is what we will score in one batch.
                IEnumerable<Candidate> candidates = allSkus.Select(sku => new Candidate {
                    CustomerId = customerId,
                    Sku = sku
                });

                // Load, encode, score
                IDataView candidateView = mlContext.Data.LoadFromEnumerable(candidates);
                IDataView candidateKeyed = encodingModel.Transform(candidateView);
                IDataView scoredView = model.Transform(candidateKeyed);

                // Pull back CustomerId, Sku, Score (raw values still present after encoding)
                List<Scored> scored = mlContext.Data
                    .CreateEnumerable<Scored>(scoredView, reuseRowObject: false)
                    .OrderByDescending(x => x.Score)
                    .Take(k)
                    .ToList();

                HashSet<int> recommendedSkus = scored.Select(x => x.Sku).ToHashSet();
                HashSet<int> actual = actualByCustomer[customerId];

                double precision = recommendedSkus.Intersect(actual).Count() / (double)k;

                totalPrecision += precision;
                userCount++;
            }

            return userCount == 0 ? 0.0 : totalPrecision / userCount;
        }

        /// <summary>
        /// Represents a candidate item associated with a specific customer and SKU.
        /// </summary>
        private sealed class Candidate {
            public int CustomerId { get; set; }
            public int Sku { get; set; }
        }

        /// <summary>
        /// Represents a scored association between a customer and a product SKU.
        /// </summary>
        /// <remarks>This type encapsulates the result of a scoring operation, typically used to indicate
        /// the relevance or affinity of a customer to a specific product. Instances of this class are commonly used in
        /// recommendation systems or analytics scenarios.</remarks>
        private sealed class Scored {
            public int CustomerId { get; set; }
            public int Sku { get; set; }
            public float Score { get; set; }
        }
        
        /// <summary>
        /// Represents an order containing customer, product, and quantity information.
        /// </summary>
        /// <remarks>This interface defines the structure for order data, Each property corresponds to a column in the input
        /// data source, as indicated by the <see cref="LoadColumnAttribute"/>.</remarks>
        public interface Order {
            [LoadColumn(0)]
            public int CustomerId { get; set; }
            [LoadColumn(1)]
            public int Sku { get; set; }
            [LoadColumn(2)]
            public float Quantity { get; set; }
        }
    }
}
