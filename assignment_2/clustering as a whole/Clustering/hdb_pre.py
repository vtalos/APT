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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

# Make sure feature matches the feature in hdb_post_cuda.py. Must be Quantity or Revenue
feature = 'Quantity'

def create_basket_string(group):
    return ' '.join([f"{subcat}_{qty}" for subcat, qty in zip(group['Subcategory'], group[feature])])

data = pd.read_csv("University Dataset 2024.csv")

required_columns = {'StoreId', 'DateTime', 'InvoiceGlobalId', 'Subcategory', 'Quantity', 'Revenue'}
data.dropna(subset=required_columns, inplace=True)

# Group data by InvoiceGlobalId and Subcategory, summing the quantities
baskets = data.groupby(['InvoiceGlobalId', 'Subcategory'])[feature].sum().reset_index()

basket_strings = baskets.groupby('InvoiceGlobalId').apply(create_basket_string).reset_index(name='Basket')

# Filter out baskets with fewer than 2 unique subcategories
basket_strings['Unique_Subcategory_Count'] = basket_strings['Basket'].apply(lambda x: len(set(x.split())))
basket_strings = basket_strings[basket_strings['Unique_Subcategory_Count'] > 2].reset_index().drop(columns=['Unique_Subcategory_Count','index'])

vectorizer = CountVectorizer(binary=False)
basket_matrix = vectorizer.fit_transform(basket_strings['Basket'])

np.save('normalized_matrix.npy', basket_matrix.todense())
basket_strings.to_csv('prepared_baskets.csv', index=False)
print("Data preparation completed. Files saved: 'normalized_matrix.npy' and 'prepared_baskets.csv'")
