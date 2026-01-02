using System;
using System.Collections.Generic;
using System.Text;

namespace Shared.Segmentation.Models {
    public sealed record Cluster(string ClusterName, int Count);
}
