using System;
using System.Collections.Generic;
using System.Text;

namespace Shared.Work {
    public enum WorkflowEventType {
        Starting = 1,
        Configuring = 2,
        Completed = 3,
        Failed = 4,
        DatasetDownloaded = 5,
        DatasetSplit = 6,
        Training = 7,
        Evaluating = 8,
        PersistingModel = 9,
        DatasetException = 10,
    };
}
