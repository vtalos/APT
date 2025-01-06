import pandas as pd
import matplotlib.pyplot as plt

# Load data from the Excel file
file_path = 'University Dataset 2024.xlsx'
df = pd.read_excel(file_path, sheet_name='Transactions')

# Convert the DateTime column to datetime type
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract day of the week and hour from the DateTime column
df['DayOfWeek'] = df['DateTime'].dt.day_name()  # Day of the week
df['Hour'] = df['DateTime'].dt.hour  # Hour

# Calculate total revenue
total_revenue = df['Revenue'].sum()

# Calculate total revenue by day of the week
daily_revenue = df.groupby('DayOfWeek')['Revenue'].sum()

# Calculate revenue percentage by day of the week
daily_revenue_percentage = (daily_revenue / total_revenue) * 100

# Print revenue percentages by day of the week
print("Revenue percentages by day of the week:")
for day, percentage in daily_revenue_percentage[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].items():
    print(f"{day}: {percentage:.2f}%")

# Create bar chart for revenue percentages by day of the week
plt.figure(figsize=(10, 6))
daily_revenue_percentage[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].plot(kind='bar', color='skyblue')
plt.title('Percentage of Total Revenue by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Revenue Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate total revenue by hour of the day
hourly_revenue = df.groupby('Hour')['Revenue'].sum()

# Calculate revenue percentage by hour of the day
hourly_revenue_percentage = (hourly_revenue / total_revenue) * 100

# Print revenue percentages by hour of the day
print("\nRevenue percentages by hour of the day:")
for hour, percentage in hourly_revenue_percentage.items():
    print(f"Hour {hour}: {percentage:.2f}%")

# Create bar chart for revenue percentages by hour of the day
plt.figure(figsize=(10, 6))
hourly_revenue_percentage.plot(kind='bar', color='lightcoral')
plt.title('Percentage of Total Revenue by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Revenue Percentage (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()