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

import pandas as pd
import numpy as np

from cuml.cluster import HDBSCAN
######################################################################
# This script must be ran in a linux conda environment where cuml must be installed. 
# This script must be ran on a system with a Nvidia GPU
# This script can't run 'as is' on windows! Use WSL and point to the right directory for the rest of the files needed
# If you want to run on windows or other non-Nvidia environments uncomment the following line and comment the one above importing from cuml
# from hdbscan import HDBSCAN
######################################################################

# Make sure feature matches the feature in hdb_pre.py. Must be Quantity or Revenue, to get the correct file name
feature = 'Quantity'

normalized_basket_matrix = np.load('normalized_matrix.npy')
basket_strings = pd.read_csv('prepared_baskets.csv')

clusterer = HDBSCAN(
    min_samples=200,
    min_cluster_size=3000,
    # cluster_selection_epsilon=1.2
)
basket_strings['Cluster'] = clusterer.fit_predict(normalized_basket_matrix)

basket_strings.to_csv(f"Clustered_Baskets_{feature}.csv", index=False)
cluster_counts = basket_strings['Cluster'].value_counts().sort_index()
num_clusters = cluster_counts[cluster_counts.index != -1].shape[0]
print(f"\nNumber of clusters (excluding noise): {num_clusters}")
print("\nCluster member counts:")
print(cluster_counts)