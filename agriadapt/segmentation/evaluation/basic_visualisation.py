from pandas import read_csv, DataFrame
from plotly import graph_objects as go


ARCHITECTURES = ["slim", "squeeze"]
WIDHTS = [25, 50, 75, 100]
SIZES = [128, 256, 512]
# METRICS = ["iou", "binaryaccuracy", "precision", "recall", "f1score", ]
METRICS = [
    "iou",
    "binaryaccuracy",
    "recall",
    "f1score",
]
REVERSE = [True]


data = read_csv("results/20240312-160851.csv")

# data = data[
#     (data["metric"] == metric)
#     # & (data["size"] == size)
#     # & (data["size"] == size)
#     & (data["reverse_training"] == reverse_training)
# ]
# data = data[(data["metric"] == metric)]

# data = data[["architecture", "width", "size", "metric", "reverse_training", "std"]]
# data["std"] = data["std"].round(4)
#
for metric in METRICS:
    print(metric)
    print("=========================")
    sub_data = data[(data["metric"] == metric)]
    x = (
        sub_data["architecture"]
        + " "
        + sub_data["width"].astype(str)
        + " "
        + sub_data["size"].astype(str)
        + " "
        + sub_data["metric"].astype(str)
    ).values
    y = sub_data["mean"].values

    for a, b in zip(x, y):
        print(a, round(b, 4))
    print("=========================")
    print()
    fig = go.Figure(data=go.Bar(x=x, y=y))
    # fig.update_layout({"yaxis_range": [0, 0.45]})
    # fig.write_image(f"results/{metric}.png")
    # fig.show()
