from torchmetrics.classification import (
    BinaryJaccardIndex,
    #BinaryPrecision,
    BinaryAccuracy,
    MulticlassAccuracy,
    BinaryRecall,
    BinaryF1Score,
)

try:
    from segmentation.local_settings import *
except:
    pass

#PROJECT_DIR = "/home/amachidon/work/agriadapt"
PROJECT_DIR = "D:\\Desktop\\GithubClone\\agriadapt-develop\\"

# SEED = 4231
SEED = 123

BATCH_SIZE = 2**1
# BATCH_SIZE = 2**2
# BATCH_SIZE = 2**3
# BATCH_SIZE = 2**4
# BATCH_SIZE = 2**5
# BATCH_SIZE = 2**6
# BATCH_SIZE = 2**7
# BATCH_SIZE = 2**8
# EPOCHS = 1000
#EPOCHS = 150
EPOCHS = 300

LEARNING_RATE = 0.0001
# LEARNING_RATE = 0.001
# LEARNING_RATE_SCHEDULER = "linear"s
LEARNING_RATE_SCHEDULER = "exponential"
# LEARNING_RATE_SCHEDULER = "no scheduler"

# MODEL = "slim"
MODEL = "squeeze"

REGULARISATION_L2 = 0.01
REGULARISATION_L2_SQUEEZE = 0.1
LEARNING_RATE_SQUEEZE = 0.0001

#REGULARISATION_L2 = 0.01
#REGULARISATION_L2_SQUEEZE = 0.001
#LEARNING_RATE_SQUEEZE = 0.00001
 
DROPOUT = 0.5
# DROPOUT = 0.75
# DROPOUT = False

# IMAGE_RESOLUTION = None
# IMAGE_RESOLUTION = (128, 128)
#IMAGE_RESOLUTION = (256, 256)
IMAGE_RESOLUTION = (512, 512)

CLASSES = [0, 1, 3, 4]
WANDB = False
WIDTHS = [0.25, 0.50, 0.75, 1.0]#.index({"knn":0.5})

KNN_WIDTHS = {
    0.25: 1,
    0.50: 5,
    0.75: 1,
}

# Infest loss weights
# LOSS_WEIGHTS = [0.1, 0.45, 0.45]
# Cofly loss weights
LOSS_WEIGHTS = [0.1, 0.9]

    
METRICS = {
    "iou": BinaryJaccardIndex,
    #"precision": BinaryPrecision,
    "precision": BinaryAccuracy,
    "recall": BinaryRecall,
    "f1score": BinaryF1Score,
}

try:
    from segmentation.local_settings import *
except:
    pass
