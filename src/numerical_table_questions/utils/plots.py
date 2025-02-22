import warnings
from datetime import datetime
from typing import List, Optional, Union

import datasets
import seaborn as sns
import matplotlib.pyplot as plt


def results_dataset_to_dataframe(dataset: Union[datasets.Dataset, str]):
    if isinstance(dataset, str):
        dataset = datasets.Dataset.load_from_disk(dataset)
    return dataset.to_pandas()


def map_model_name(model_name: str) -> str:
    match model_name.lower():
        case s if 'llama' in s:
            return 'Llama-3.3'
        case s if 'qwen' in s:
            return 'Qwen-2.5'
        case s if 'phi' in s:
            return 'Phi-4'
        case s if 'gemma' in s:
            return 'Gemma-2'
        case s if 'tapas' in s:
            return 'TAPAS'
        case s if 'tapex' in s:
            return 'TAPEX'
        case s if 'omnitab' in s:
            return 'OmniTab'
        case s if 'reast' in s:
            return 'ReasTAP'
        case s if 'sqlcoder' in s:
            return 'SQLCoder'
        case _:
            return model_name


def process_xtick_labels(x_col: str, current_labels: List[str]) -> List[str]:
    match x_col:
        case 'aggregation_num_rows' | 'boundary':
            new_text = [
                f"<{current_labels[i].get_text()}" if i == 0
                else (
                    f"{current_labels[i].get_text()}-{current_labels[i+1].get_text()}" if i < (len(current_labels)-1)
                    else f">{current_labels[i].get_text()}"
                    )
                for i in range(len(current_labels))
                ]
            for text, new in zip(current_labels, new_text):
                text.set_text(new)
            return current_labels
        case 'aggregator':
            for text in current_labels:
                if text.get_text().strip() == '':
                    text.set_text('noop')
            return current_labels
        case 'condition':
            for text in current_labels:
                if text.get_text().strip() == '':
                    text.set_text('none')
            return current_labels
        case _:
            new_text = [label.get_text().lower().replace('_', ' ') for label in current_labels]
            for text, new in zip(current_labels, new_text):
                text.set_text(new)
            return current_labels


def grouped_plot(data, x="boundary", y="exact_match", hue="model_name", kind='bar', sve_path: Optional[str] = None, axes_args: dict = {}, color_palette: str = 'colorblind', style: str = 'whitegrid', context: str = 'paper'):
    sns.set_palette(color_palette)
    sns.set_style(style)
    sns.set_context(context)
    if kind == 'line':
        ax = sns.lineplot(data, x=x, y=y, hue=hue)
        legend = ax.get_legend() or ax.legend()
    elif 'hist' in kind:
        ax = sns.histplot(data, x=x)
        legend = ax.get_legend() or ax.legend()
    elif 'bar' in kind:
        ax = sns.barplot(data, x=x, y=y, hue=hue)
        legend = ax.get_legend() or ax.legend()
    else:
        warnings.warn("Using generic catplot, this can lead to problems with legend placement because the plot is figure level not axes level.")
        facet_grid = sns.catplot(data, x=x, y=y, hue=hue, kind=kind)
        ax = facet_grid.ax
        legend = facet_grid._legend
    ax.set(**axes_args)
    current_xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(process_xtick_labels(x, current_xticklabels))
    # fix legend labels (model names)
    new_labels = [map_model_name(label.get_text()) for label in legend.texts]
    for t, l in zip(legend.texts, new_labels):
        t.set_text(l)
    legend.set(title="Model Name" if hue == 'model_name' else hue.strip().replace('_', ' '), loc='best')
    plt.savefig(sve_path if sve_path is not None else './plots/' + f"plot_{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}.pdf")
    plt.clf()
