import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles


def load_data(file_name, separator=';'):
    """
    Function to load *.csv data.csv
    :param file_name: String
    :param separator: E.g., ',', ';'
    :return: DataFrame
    """
    try:
        dataDF = pd.read_csv(file_name, sep=separator)
        print('FILE EXIST')
        return dataDF
    except IOError as ioe:
        # file didn't exist (or other issues)
        print('File does not exist!')
        print(ioe)
        return False


def category_percentage(df, categories):
    """
    Function to visualize the label or category distribution in the dataset.
    :param df: DataFrame
    :param categories: List
    :return:
    """
    plot_data = df[categories].mean() * 100

    plt.figure(figsize=(10, 5))
    plt.title('Percentage of Samples per Category')
    sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.xticks(rotation=45)
    plt.show()
    return


def correlation_between_labels(df):
    plt.figure(figsize=(15, 8))
    plt.title("Correlation between the different categories")
    sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
    plt.show()
    return


def venn_diagram(df, categories, G1, G2, G3, G4):
    """
    Functions for plotting area-proportional three-way Venn diagram. Intersection between categories.
    G1, G2, G3, and G4, each should contain up to three indices corresponding to the labels that will be
     intercepted in each diagram. This functions returns four Venn diagrams.
    :param df: Pandas DataFrame
    :param categories: List
    :param G1: Array of int
    :param G2: Array of int
    :param G3: Array of int
    :param G4: Array of int
    :return:
    """
    figure, axes = plt.subplots(2, 2, figsize=(20, 20))

    labels = {label: set(df[df[label] == 1].index) for label in categories[:-1]}

    v1 = venn3([labels[categories[index]] for index in G1],
               set_labels=(categories[index] for index in G1), set_colors=('#a5e6ff', '#3c8492', '#9D8189'),
               ax=axes[0][0])
    for text in v1.set_labels:
        text.set_fontsize(22)

    v2 = venn3([labels[categories[index]] for index in G2],
               set_labels=(categories[index] for index in G2), set_colors=('#e196ce', '#F29CB7', '#3c81a9'),
               ax=axes[0][1])
    for text in v2.set_labels:
        text.set_fontsize(22)

    v3 = venn3([labels[categories[index]] for index in G3],
               set_labels=(categories[index] for index in G3), set_colors=('#a5e6ff', '#F29CB7', '#9D8189'), ax=axes[1][0])
    for text in v3.set_labels:
        text.set_fontsize(22)

    v4 = venn3([labels[categories[index]] for index in G4],
               set_labels=(categories[index] for index in G4), set_colors=('#e196ce', '#3c81a9', '#9D8189'),
               ax=axes[1][1])
    for text in v4.set_labels:
        text.set_fontsize(22)
    plt.show()
    return