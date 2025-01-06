import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Reading the data
df = pd.read_excel('University Dataset 2024.xlsx', sheet_name='Transactions')

# Calculating product frequency
product_freq = df['Subcategory'].value_counts()
very_frequent_products = set(product_freq[product_freq > product_freq.mean()].index)

# Creating pivot table and converting to boolean
basket = pd.crosstab(df['InvoiceGlobalId'], df['Subcategory'])
basket = basket.astype(bool)

# Applying the apriori algorithm with very low support
frequent_itemsets = apriori(basket, 
                          min_support=0.001,  # 0.1% minimum support
                          use_colnames=True)

# Creating the association rules with very low confidence
rules = association_rules(frequent_itemsets, 
                        metric="confidence",
                        min_threshold=0.01,  # 1% minimum confidence
                        num_itemsets=None)

# Adding the size of the sets as columns
rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
rules['total_items'] = rules['antecedent_len'] + rules['consequent_len']

# Filtering to keep more interesting rules
def is_interesting_rule(row):
    antecedents = set(row['antecedents'])
    consequents = set(row['consequents'])
    all_items = antecedents.union(consequents)
    
    # At least one product should not be among the very frequent ones
    has_non_frequent = any(item not in very_frequent_products for item in all_items)
    
    # No more than 3 products in total
    good_size = row['total_items'] <= 3
    
    return has_non_frequent and good_size

# Applying the filter
interesting_rules = rules[rules.apply(is_interesting_rule, axis=1)].copy()

# Adding randomness to the sorting for variety
interesting_rules['random_factor'] = np.random.uniform(0, 1, size=len(interesting_rules))
interesting_rules['adjusted_lift'] = interesting_rules['lift'] * interesting_rules['random_factor']

# Sorting based on the adjusted lift
interesting_rules = interesting_rules.sort_values('adjusted_lift', ascending=False)

# Converting to percentages and rounding
interesting_rules['support'] = interesting_rules['support'] * 100
interesting_rules['confidence'] = interesting_rules['confidence'] * 100
interesting_rules = interesting_rules.round(2)

def print_rules(rules):
    if len(rules) == 0:
        print("No rules found with the current criteria.")
        return
        
    for idx, rule in rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        print(f"Rule {idx+1}:")
        print(f"IF customer buys: {antecedents}")
        print(f"THEN they're likely to buy: {consequents}")
        print(f"Support: {rule['support']}%")
        print(f"Confidence: {rule['confidence']}%")
        print(f"Lift: {rule['lift']}")
        print("-" * 50)

# Displaying the rules
print("\nInteresting Association Rules:")
print_rules(interesting_rules.head(20))