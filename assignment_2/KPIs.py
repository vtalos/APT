import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = "University Dataset 2024.xlsx"
transactions = pd.read_excel(file_path, sheet_name="Transactions")
stores = pd.read_excel(file_path, sheet_name="Stores")
categories = pd.read_excel(file_path, sheet_name="Categories")

# Merge datasets for analysis
transactions = transactions.merge(stores, on="StoreId", how="left")
transactions = transactions.merge(categories, on="Subcategory", how="left")

# KPI Calculations
results = {}

# 1. Total Revenue
total_revenue = transactions["Revenue"].sum()
results["Total Revenue"] = [f"${total_revenue:,.2f}"]

# 2. Revenue by Store
revenue_by_store = transactions.groupby("StoreId")["Revenue"].sum().sort_values(ascending=False)
results["Revenue by Store"] = revenue_by_store

# 3. Revenue by Category
revenue_by_category = transactions.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
results["Revenue by Category"] = revenue_by_category

# 4. Revenue by Category and Megacategory
revenue_by_category_megacategory = transactions.groupby(["Megacategory", "Category"])["Revenue"].sum().sort_values(ascending=False)
results["Revenue by Cat-Mega"] = revenue_by_category_megacategory

# 5. Top-Selling Products by Category
top_selling_by_category = transactions.groupby(["Category", "Subcategory"])["Quantity"].sum().sort_values(ascending=False)
results["Top-Selling by Cat"] = top_selling_by_category

# 6. Underperforming Products (Lowest Revenue)
underperforming_products = transactions.groupby("Subcategory")["Revenue"].sum().sort_values().head(5)
results["Underperforming"] = underperforming_products

# 7. Average Basket Size
total_quantity = transactions["Quantity"].sum()
total_transactions = transactions["InvoiceGlobalId"].nunique()
average_basket_size = total_quantity / total_transactions
results["Avg Basket Size"] = [f"{average_basket_size:.2f} items"]

# 8. Average Basket Value
average_basket_value = total_revenue / total_transactions
results["Avg Basket Value"] = [f"${average_basket_value:,.2f}"]

# 9. Average Basket Value per Store Type
average_basket_value_store_type = transactions.groupby("StoreType").apply(
    lambda x: x["Revenue"].sum() / x["InvoiceGlobalId"].nunique()
).sort_values(ascending=False)
results["Avg Basket Val by Type"] = average_basket_value_store_type

# 10. Average Basket Size per Store Type
average_basket_size_store_type = transactions.groupby("StoreType").apply(
    lambda x: x["Quantity"].sum() / x["InvoiceGlobalId"].nunique()
).sort_values(ascending=False)
results["Avg Basket Size by Type"] = average_basket_size_store_type

# 11. Top-Selling Categories per Store Type
top_selling_categories_store_type = transactions.groupby(["StoreType", "Category"])["Quantity"].sum().sort_values(ascending=False)
results["Top-Selling Cat by Type"] = top_selling_categories_store_type

# 12. Sales Volume by Store Type
sales_volume_by_store_type = transactions.groupby("StoreType")["Quantity"].sum().sort_values(ascending=False)
results["Sales Vol by Type"] = sales_volume_by_store_type

# 13. Revenue per Unit Sold (by Subcategory)
revenue_per_unit = transactions.groupby("Subcategory").apply(
    lambda x: x["Revenue"].sum() / x["Quantity"].sum()
).sort_values(ascending=False)
results["Rev per Unit"] = revenue_per_unit

# 14. Top-Selling Products by Quantity
top_selling_products = transactions.groupby("Subcategory")["Quantity"].sum().sort_values(ascending=False)
results["Top-Selling Products"] = top_selling_products

# 15. Top-Selling Products by Store Type (Top 5)
top_5_products_by_store_type = transactions.groupby(["StoreType", "Subcategory"])["Quantity"].sum().sort_values(ascending=False).groupby(level=0).head(5)
results["Top 5 Prod by Type"] = top_5_products_by_store_type

# 16. Most Underperforming Products by Store Type (Bottom 5 by Revenue)
bottom_5_products_by_store_type = transactions.groupby(["StoreType", "Subcategory"])["Revenue"].sum().sort_values().groupby(level=0).head(5)
results["Bottom 5 Prod by Type"] = bottom_5_products_by_store_type

# Save results to an Excel file
output_file = "KPI_Results.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, data in results.items():
        if isinstance(data, pd.Series):
            data.to_frame(name="Value").to_excel(writer, sheet_name=sheet_name[:31])  # Short sheet names
        elif isinstance(data, list):
            pd.DataFrame(data, columns=["Value"]).to_excel(writer, sheet_name=sheet_name[:31], index=False)

print(f"Results have been saved to '{output_file}'.")