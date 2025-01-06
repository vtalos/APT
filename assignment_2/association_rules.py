import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Reading the data
df = pd.read_excel('University Dataset 2024.xlsx', sheet_name='Transactions')

# Creating pivot table and converting to boolean
basket = pd.crosstab(df['InvoiceGlobalId'], df['Subcategory'])
basket = basket.astype(bool)

# Applying the apriori algorithm
frequent_itemsets = apriori(basket, 
                          min_support=0.005,  # 0.5% minimum support
                          use_colnames=True)

# Creating the association rules
rules = association_rules(frequent_itemsets, 
                        metric="confidence",
                        min_threshold=0.05,  # 5% minimum confidence
                        num_itemsets=None)

# Sorting the rules by lift
rules = rules.sort_values('lift', ascending=False)

# Adding the support % as a percentage
rules['support'] = rules['support'] * 100
rules['confidence'] = rules['confidence'] * 100

# Rounding the numbers
rules = rules.round(2)

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

# Displaying more rules
print("\nTop Association Rules:")
print_rules(rules.head(15))

# Displaying statistics for product categories
print("\nProduct Categories Frequency:")
product_freq = df['Subcategory'].value_counts()
print(product_freq)

# Displaying summary statistics for the rules
print("\nSummary Statistics for Rules:")
print(f"Total number of rules found: {len(rules)}")
print("\nLift Statistics:")
print(rules['lift'].describe())
print("\nConfidence Statistics:")
print(rules['confidence'].describe())
print("\nSupport Statistics:")
print(rules['support'].describe())