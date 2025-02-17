from datetime import datetime
from typing import Optional, Union

import datasets
import seaborn as sns
import matplotlib.pyplot as plt


def results_dataset_to_dataframe(dataset: Union[datasets.Dataset, str]):
    if isinstance(dataset, str):
        dataset = datasets.Dataset.load_from_disk(dataset)
    return dataset.to_pandas()


def grouped_plot(data, x="boundary", y="exact_match", hue="model_name", kind='bar', sve_path: Optional[str] = None, color_palette='colorblind'):
    sns.set_palette(color_palette)
    if kind == 'line':
        ax = sns.lineplot(data, x=x, y=y, hue=hue)
    else:
        ax = sns.catplot(data, x=x, y=y, hue=hue, kind=kind)
    ax.figure.savefig(sve_path if sve_path is not None else './plots/' + f"plot_{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}.pdf")
    plt.clf()
