# DOS — Workflow Abstractions
A lightweight, open-source workflow contract layer built in C# on top of ML.NET principles.
This package provides interfaces and data records only — no internal training recipes or infrastructure — making it safe to share while still giving developers powerful building blocks to orchestrate their own machine learning pipelines.

![NuGet Version](https://img.shields.io/nuget/v/ReadyDOS.ML.Shared?style=flat)

## Getting Started
I personally use `IWorkflow` in my own projects and real production-style prototypes, and I plan to adapt it soon for high-quality segmentation visualizations.

------------------------
## + OCR & PDF AI Workflow implementatnions 
## 🚀 Features Overview
- [x] Reads unstructured or scanned PDFs (contracts, filings, policies, claims, regulations)
- [x] Extracts text via embedded content or OCR, normalizes it, featurizes it, clusters it
- [x] Produces quantifiable training manifests and preview-scored results
- [x] Supports automated retraining, quality comparison, and best-model promotion
- [x] Search & Replace
- [x] Could runs on scalable worker services (CPU or GPU), suitable for ML-Ops or regulated intelligence pipelines
- [x] Cloud storage integration


### 📝 Text Formatting
- **AI-powered legal documentation processing** at scale
- Automates extraction and normalization of dense legal text from PDFs, contracts, filings, and statutes
- Enables semantic clustering of documents for compliance review, e-discovery, risk triage, and regulatory audit readiness
- Converts unstructured legal language into high-quality ML feature vectors for downstream analytics or model retraining
- Reduces manual review costs, accelerates case preparation, and improves accuracy by eliminating human transcription error
- Supports conditional dependency pathways (embedded text vs OCR) to maximize recall even on scanned or poor-quality legal sources
- Generates preview-scored cluster assignments for rapid expert validation (e.g., grouping similar clauses or precedent families)
- Best for: Law firms, compliance teams, e-discovery vendors, policy analysts, and regulated enterprises

------------------------------------------------------
* 12/25 Update
* End-to-end ML workflow orchestration (data readiness → training → evaluation → persistence)
* Structured metadata for dataset lineage and splits (DataObjectInfo, DataSplitInfo)
* Consistent UTC timestamps for distributed and serverless systems
* Algorithm identity mapping for business-friendly reporting
* Clean log transport that integrates with any ILogger via IProgress

## What the Shared Library Provides
*  🧠 Compose end-to-end ML workflows
*  *(Data preparation → Training → Evaluation → Model Selection → Persistence)*
* 📦 Generic workflow signatures that can cluster or train any dataset
* 🧬 Dataset lineage tracking via metadata records:
* `DataObjectInfo` → dataset source + last modified time
* `DataSplitInfo` → train/eval dataset handles + row counts + origin key
* 🕒 Consistent UTC timestamps for distributed and serverless systems
* 🧾 Public-safe log transport, designed to integrate with any ILogger using:
* 📊 Business-aligned KPI language in workflow results for revenue impact
* 🧩 Algorithm identity mapping without exposing internal ML pipelines
* 🚀 Scalable orchestration, ready for APIs, background workers, or dashboards

| **Currently Supported Algorithms** | **Display Name**  | **Examples**   
| ----------------------------- | ---------------------- | ---------------------------------------------------------- |
| Matrix Factorization          | Product Recommendations, Basket Expansion       | Increase Average Order Value (AOV) and purchase frequency. Recommendations to increase basket size                    |
| L-BFGS Optimization           | Retention              | High-performance linear churn or propensity classification. Automate discounts in batches for customers at risk of churning, asn an example.|
| KMeans++ Clustering           | Customer Segmentation  | Beheavioural insights into your customers/users.                        |
| LightGBM                      | Forecasting			 | Sales, inventory demand/trends, revenue forecasting.                  |

