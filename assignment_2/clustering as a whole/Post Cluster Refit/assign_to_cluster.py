# Copyright (c) 2025 Panagiotis Alexandros Daskalopoulos
# Author: Panagiotis Alexandros Daskalopoulos
# Contact: panosdaskalopoulos259@gmail.com
#
# All rights reserved.
#
# Permission is granted to use, view, and study this software for personal or educational purposes.
# Redistribution or modification of this software, in whole or in part, in any form or by any means,
# is prohibited without explicit, prior written consent from the copyright holder.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
######################################################################
# This script takes the unclustered baskets and the representative clusters
# and tries to fit the unclustered basket to the known representative clustered ones
# This is done by set subtraction and counting missing elements
######################################################################

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np


def create_cluster_category_dict(file_path):

    df: DataFrame = pd.read_csv(file_path)
    cluster_dict = {}
    cluster_set = set()

    for _, row in df.iterrows():
        cluster = int(row['Cluster'])

        cluster_set.add(cluster)

        cat1 = row['Category_1'].strip()
        cat2 = row['Category_2'].strip() if pd.notna(row['Category_2']) else ''

        if cat2:
            categories = {cat1, cat2}
        else:
            categories = {cat1}
        cluster_dict[cluster] = categories

    return cluster_dict, cluster_set


def read_unclustered_baskets(file: str) -> set[str]:
    df = pd.read_csv(file)
    invoices = set(df["InvoiceGlobalId"].to_list())
    return invoices


def convert_to_set(series: Series) -> set[str]:
    """
    Takes a series of a Pandas Dataframe and returns the values as a set

    :param series: Series from pandas column with strings as contents
    :returns: The series values as a set
    """
    final_set = set(series.values.tolist())
    return final_set


def read_selected_basket_data(file: str, invoices: set[str]) -> dict[str, set[str]]:
    """
    Takes the dataframe with transaction ids and keeps only the transactions found in the invoices set.
    Then groups by InvoiceGlobalId keeping only the InvoiceGlobalId and Subcategory converted to a set
    with each string as a item in that set. Retuns that as a dictionairy
    """
    data = pd.read_csv(file)
    data = data[data['InvoiceGlobalId'].isin(invoices)]
    required_columns = {'StoreId', 'DateTime', 'InvoiceGlobalId', 'Subcategory', 'Quantity', 'Revenue'}
    data.dropna(subset=required_columns, inplace=True)

    # Group data by InvoiceGlobalId and Subcategory, summing the quantities
    baskets: DataFrame = data.groupby(['InvoiceGlobalId', 'Subcategory'])["Quantity"].sum().reset_index()
    baskets = baskets.groupby("InvoiceGlobalId")['Subcategory'].apply(convert_to_set)
    return baskets.to_dict()


def get_top_5_items(cluster_dict: dict):
    return sorted(cluster_dict.items(), key=lambda x: x[1], reverse=True)[:5]


def results_and_plot(result_dict: dict):

    for cluster, items in result_dict.items():
        top_5_items = get_top_5_items(items)

        print(f"Cluster {cluster}:")
        for item, value in top_5_items:
            print(f"  {item}: {value}")

        items_list, values_list = zip(*top_5_items)

        plt.figure(figsize=(10, 5))
        plt.bar(items_list, values_list, color=colormaps.get_cmap('Dark2')(np.linspace(0, 1, len(top_5_items))))
        plt.title(f'Top 5 Items in Cluster {cluster}')
        plt.xlabel('Items')
        plt.ylabel('Values')
        plt.show()


if __name__ == "__main__":
    clusters_dict, cluster_set = create_cluster_category_dict("Representative_Clusters.csv")
    print(cluster_set)
    unclustered_invoices = read_unclustered_baskets("Unclustered.csv")
    basket_data = read_selected_basket_data("University Dataset 2024.csv", unclustered_invoices)
    result_dict = dict()

    # ------------------------------------------ dict initialization start ------------------------------------------ #
    upsell_dict_cluster = {}
    upsell_dict_invoice = {}

    for cluster in cluster_set:
        upsell_dict_cluster[cluster]: int = {}

    for unclustered_invoice in unclustered_invoices:
        upsell_dict_invoice[unclustered_invoice]: str = {}
    # ------------------------------------------- dict initialization end ------------------------------------------- #

    # This could have been a hand compiled set or list, but I want the script to be category agnostic
    global_category_set = set()

    # Finds the differences between clusters modeled basket and non-clustered actual baskets
    # If the model basket is not a subset of the unclustered basket we use an empty set
    # As this empty set is never going to be ever used we set it using {*()} instead of set()
    # as it's faster (no global call)
    for cluster in cluster_set:
        current_basket: set = clusters_dict[cluster]
        global_category_set: set[str] = global_category_set.union(current_basket)
        for unclustered_invoice in unclustered_invoices:
            global_category_set: set[str] = global_category_set.union(basket_data[unclustered_invoice])
            if current_basket.issubset(basket_data[unclustered_invoice]):
                upsell_dict_cluster[cluster][unclustered_invoice]: set = basket_data[unclustered_invoice]\
                    .difference(current_basket)
            else:
                upsell_dict_cluster[cluster][unclustered_invoice]: set = {*()}

    for cluster in cluster_set:
        result_dict[cluster] = dict()
        for item in global_category_set:
            result_dict[cluster][item] = 0
        for invoice in unclustered_invoices:
            for item_bought in upsell_dict_cluster[cluster][invoice]:
                result_dict[cluster][item_bought] += 1

    results_and_plot(result_dict)
