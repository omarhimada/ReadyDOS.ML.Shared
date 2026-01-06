# ReadyDOS — ML Workflows & Abstractions
AI/ML Workflow interfaces, implementations, and extensions written in C#. 
Giving developers powerful building blocks to orchestrate machine learning pipelines.

![Build Status](https://github.com/omarhimada/ReadyDOS.ML.Shared/actions/workflows/readydos-shared-nuget-deploy.yml/badge.svg)
![NuGet Version](https://img.shields.io/nuget/v/ReadyDOS.ML.Shared?style=flat)

There is currently a working proof-of-concept here: [AdosiML.com](https://adosiml.com)

- `IWorkflow` for scheduling end-to-end ML workflows. Data ingestion, featurization, training, evaluation, and model persistence steps.
- Track dataset lineage, regression & classificiation model metrics, data splits, and other metadata.
- Extended `KMeans++` clustering output, where the segments are given priorities - attaching actionable business intelligence
- Included normalization methods to ensure adaptability to any dataset, organization or business. 
- The produced `Prioritized Normalized Segmentation` can be integrated into your own organizations/businesses for actionable intelligence, value automated processes or manual intervention.
  - It also includes properties that make it easy to visualize with common UI libraries.
  - *e.g.: deserialize to `JSON` with your API and it is adapatable for producing chart components with many front-end frameworks compatible with Angular, TypeScript React, Blazor, WebAssembly, anything*.

## What is provided?
* I'm happy to give developers building blocks to orchestrate their own machine learning pipelines,
  * Sharing certain parts of my consulting company `ReadyDOS`.

## What could I do with it?
  * Compose and orchestrate end-to-end ML `IWorkflow` with your integrations
    * *Data preparation → Training → Evaluation → Model Selection → Persistence*
    * `ReadyDOS.ML.Shared` is designed with scalable orchestration in mind, in a cloud-agnostic way
    * For example:
      * Use singleton worker processes to schedule a concrete `IWorkflow` in a container, persisting the models in `S3` or `Azure Blob Storage`
      * Orchestrate with containerization and CI/CD to `ECS Fargate`, `Kubernetes`, `Elastic Beanstalk`, `Azure App Service`, whatever your client business wants
      * Then, implement an application layer to load the trained model using `API Gateway + Lambda` or `Azure Functions`, or whatever, to make predictions or inferences remotely using `HTTP/gRPC` etc.

### :heavy_plus_sign: New `PrioritizedNormalizedSegmentation`
- 💹 Features Overview
  - Extends typical RFM clustering (recency, frequency, monetary) with
    - Prioritization of segments
    - Normalization to ensure accessibility across organizations and business's different datasets
  - Output easily formatted for dashboard visualizations with whatever UI front-end your application is using


## Update on ReadyDOS
| **Currently Implemented **                   |  :globe_with_meridians: Example Use Cases                                             |
| ---------------------------------------------|---------------------------------------------------------------------------------------|
| Matrix Factorization                         | Customer/user-specific recommendations                                                |
| L-BFGS Optimization                          | Churn prediction, supporting the ability to do fraud detection, and sales forecasting | 
| KMeans++ Clustering                          | Provide actionable insight with segmentation (VIP members, dormant value, etc.)       |

<img width="3330" height="1816" alt="readydos3x1" src="https://github.com/user-attachments/assets/4b8998fe-8112-487f-9d89-eb9b995aaa55" />
https://adosiml.com


