import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def remove_outliers(df, columns, n_std=3):
    """
    Remove outliers using z-score
    """
    df_clean = df.copy()
    
    for column in columns:
        # Calculate z-score
        z_scores = stats.zscore(df_clean[column])
        # Keep only rows with z-score less than n_std
        df_clean = df_clean[abs(z_scores) < n_std]
    
    return df_clean

def load_and_prepare_data(pos_file):
    # Read the Excel sheets
    pos_data = pd.read_excel(pos_file, sheet_name='POS Data')
    loyalty_data = pd.read_excel(pos_file, sheet_name='Loyalty')
    hierarchy_data = pd.read_excel(pos_file, sheet_name='Hierarchy Categories & Barcodes')
    
    # Merge the data
    data = pos_data.merge(hierarchy_data, on='Barcode', how='left')
    data = data.merge(loyalty_data, left_on='LoyaltyCard_ID', right_on='Cardholder', how='left')
    
    # Convert date to datetime
    data['Date_'] = pd.to_datetime(data['Date_'])
    
    return data, loyalty_data

def create_customer_metrics(data):
    # 1. Basic metrics per customer
    customer_metrics = data.groupby('LoyaltyCard_ID').agg({
        'Basket_ID': 'nunique',  # Number of unique visits
        'Value_': 'sum',         # Total purchase value
        'Quantity': 'sum',       # Total number of products
        'Date_': ['nunique', 'min', 'max']  # Date statistics
    }).reset_index()
    
    # Rename columns
    customer_metrics.columns = ['LoyaltyCard_ID', 'total_visits', 'total_spend', 
                              'total_items', 'unique_days', 'first_purchase', 'last_purchase']
    
    # 2. Calculate additional metrics
    customer_metrics['avg_basket_size'] = customer_metrics['total_spend'] / customer_metrics['total_visits']
    customer_metrics['avg_item_value'] = customer_metrics['total_spend'] / customer_metrics['total_items']
    customer_metrics['visit_frequency'] = customer_metrics['unique_days'] / customer_metrics['total_visits']
    
    # Calculate customer lifetime in days
    customer_metrics['customer_lifetime_days'] = (customer_metrics['last_purchase'] - 
                                                customer_metrics['first_purchase']).dt.days + 1  # Add 1 to avoid 0
    
    # Calculate average days between purchases
    customer_metrics['avg_days_between_purchases'] = customer_metrics['customer_lifetime_days'] / customer_metrics['total_visits']
    
    # 3. Product categories per customer
    category_pivot = pd.pivot_table(
        data, 
        values='Quantity',
        index='LoyaltyCard_ID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate percentages per category
    category_percentages = category_pivot.div(category_pivot.sum(axis=1), axis=0)
    
    # Merge all metrics
    final_metrics = customer_metrics.merge(
        category_percentages, 
        left_on='LoyaltyCard_ID', 
        right_index=True,
        how='left'
    )
    
    return final_metrics

def find_optimal_k(metrics_df, feature_columns, max_k=10):
    """
    Use the elbow method to find the optimal number of clusters.
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    wcss = []
    
    # Scale the features
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(metrics_df[feature_columns])
    
    # Calculate WCSS for different values of k
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)  # WCSS is stored in kmeans.inertia_
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()
    
    return wcss

def perform_clustering(metrics_df, feature_columns=None):
    """
    Performs clustering analysis with optimal k selection using elbow method
    """
    if feature_columns is None:
        feature_columns = ['total_spend', 'total_visits', 'avg_basket_size', 
                         'visit_frequency', 'avg_days_between_purchases']
    
    # Remove outliers
    metrics_clean = remove_outliers(metrics_df, feature_columns)
    
    # Find optimal k
    print("Determining optimal number of clusters...")
    wcss = find_optimal_k(metrics_clean, feature_columns, max_k=10)
    
    # Calculate the elbow using the second derivative
    k_values = range(1, len(wcss) + 1)
    second_derivative = np.diff(np.diff(wcss))
    optimal_k = k_values[np.argmax(second_derivative) + 1]
    
    print(f"Optimal number of clusters determined: {optimal_k}")
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(metrics_clean[feature_columns])
    
    # Perform K-means clustering with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add clusters to the dataframe
    metrics_clean['Cluster'] = clusters
    
    return metrics_clean, kmeans.cluster_centers_, optimal_k

def analyze_segments(segmented_df):
    # Calculate basic statistics per cluster
    segment_stats = segmented_df.groupby('Cluster').agg({
        'LoyaltyCard_ID': 'count',
        'total_spend': ['mean', 'median', 'std'],
        'total_visits': ['mean', 'median'],
        'avg_basket_size': ['mean', 'median'],
        'avg_days_between_purchases': ['mean', 'median'],
        'total_items': ['mean', 'median']
    })
    
    # Calculate customer percentages per cluster
    total_customers = len(segmented_df)
    segment_stats['customer_percentage'] = (segment_stats[('LoyaltyCard_ID', 'count')] / total_customers * 100).round(2)
    
    # Calculate average percentages per product category
    category_columns = [col for col in segmented_df.columns if col not in [
        'LoyaltyCard_ID', 'Cluster', 'total_spend', 'total_visits', 'avg_basket_size',
        'visit_frequency', 'avg_days_between_purchases', 'total_items', 'unique_days',
        'first_purchase', 'last_purchase', 'customer_lifetime_days', 'avg_item_value'
    ]]
    
    category_stats = segmented_df.groupby('Cluster')[category_columns].mean()
    
    return segment_stats, category_stats

def plot_segment_analysis(segmented_df, analysis):
    # Remove the problematic style setting
    # Instead, we'll use basic matplotlib styling with grid
    plt.rcParams['figure.figsize'] = (20, 15)
    plt.rcParams['axes.grid'] = True
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Distributions of key metrics per cluster', fontsize=16)
    
    # Total spend distribution
    sns.boxplot(x='Cluster', y='total_spend', data=segmented_df, ax=axes[0,0])
    axes[0,0].set_title('Total spend per cluster')
    axes[0,0].set_ylabel('Total spend (€)')
    
    # Average basket size distribution
    sns.boxplot(x='Cluster', y='avg_basket_size', data=segmented_df, ax=axes[0,1])
    axes[0,1].set_title('Average basket size per cluster')
    axes[0,1].set_ylabel('Average basket size (€)')
    
    # Visit frequency distribution
    sns.boxplot(x='Cluster', y='visit_frequency', data=segmented_df, ax=axes[1,0])
    axes[1,0].set_title('Visit frequency per cluster')
    axes[1,0].set_ylabel('Visit frequency')
    
    # Average days between purchases distribution
    sns.boxplot(x='Cluster', y='avg_days_between_purchases', data=segmented_df, ax=axes[1,1])
    axes[1,1].set_title('Average days between purchases per cluster')
    axes[1,1].set_ylabel('Days')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        segmented_df['total_visits'],
        segmented_df['total_spend'],
        c=segmented_df['Cluster'],
        s=segmented_df['avg_basket_size']*20,
        alpha=0.6,
        cmap='viridis'
    )
    plt.xlabel('Total visits')
    plt.ylabel('Total spend (€)')
    plt.title('Customer clusters\n(Circle size = Average basket size)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.show()


def characterize_segments(segment_stats, category_stats):
    """Create descriptive characterizations for each cluster"""
    descriptions = {}
    
    for cluster in segment_stats.index:
        # Basic statistics
        size = segment_stats.loc[cluster, ('LoyaltyCard_ID', 'count')]
        pct = float(segment_stats.loc[cluster, 'customer_percentage'])  # Convert to float
        avg_spend = float(segment_stats.loc[cluster, ('total_spend', 'mean')])
        avg_basket = float(segment_stats.loc[cluster, ('avg_basket_size', 'mean')])
        avg_visits = float(segment_stats.loc[cluster, ('total_visits', 'mean')])
        avg_days = float(segment_stats.loc[cluster, ('avg_days_between_purchases', 'mean')])
        
        # Find dominant categories
        top_categories = category_stats.loc[cluster].nlargest(6)
        
        description = f"""
        Cluster {cluster}:
        - Size: {int(size)} customers ({pct:.1f}% of total)
        - Average total spend: {avg_spend:.2f}€
        - Average basket size: {avg_basket:.2f}€
        - Average number of visits: {avg_visits:.1f}
        - Average days between purchases: {avg_days:.1f}
        
        Dominant product categories:
        """
        
        for cat, pct_value in top_categories.items():
            description += f"  - {cat}: {float(pct_value)*100:.1f}%\n"
        
        descriptions[cluster] = description.strip()
    
    return descriptions

def analyze_loyalty_status(segmented_customers, loyalty_data):
    """
    Analyzes the loyalty status distribution per cluster.
    """
    # Merge segmented customers with loyalty data
    loyalty_cluster = segmented_customers.merge(
        loyalty_data, 
        left_on='LoyaltyCard_ID', 
        right_on='Cardholder', 
        how='left'
    )
    
    # Create a crosstab to calculate the count of statuses per cluster
    status_dist = pd.crosstab(loyalty_cluster['Cluster'], loyalty_cluster['Status'])
    
    # Normalize the crosstab to compute percentages within each cluster
    status_percentages = status_dist.div(status_dist.sum(axis=1), axis=0) * 100
    
    # Add the normalized percentages to the original crosstab
    status_analysis = status_dist.copy()
    for column in status_dist.columns:
        status_analysis[column + ' (%)'] = status_percentages[column].round(2)
    
    return status_analysis


def main(file_path):
    # Load and prepare data
    data, loyalty_data = load_and_prepare_data(file_path)
    
    # Create customer metrics
    customer_metrics = create_customer_metrics(data)
    
    # Define features for clustering
    feature_columns = ['total_spend', 'total_visits', 'avg_basket_size', 
                      'visit_frequency', 'avg_days_between_purchases']
    
    # Perform clustering with automatic optimal k selection
    segmented_customers, cluster_centers, optimal_k = perform_clustering(
        customer_metrics, 
        feature_columns
    )
    
    # Analyze the segments
    segment_stats, category_stats = analyze_segments(segmented_customers)
    
    # Analyze loyalty status
    loyalty_status_analysis = analyze_loyalty_status(segmented_customers, loyalty_data)
    
    # Visualize the segments
    plot_segment_analysis(segmented_customers, segment_stats)
    
    # Characterize the segments
    segment_descriptions = characterize_segments(segment_stats, category_stats)
    
    return (segmented_customers, segment_stats, category_stats, 
            segment_descriptions, loyalty_status_analysis, optimal_k)

# Execute the main flow
file_path = 'POS_DATA_BAPT_2023_updated.xlsx'
(segments, analysis, category_analysis, 
 descriptions, loyalty_analysis, optimal_k) = main(file_path)

print(f"\nOptimal number of clusters: {optimal_k}")

# Display segment descriptions
for cluster, description in descriptions.items():
    print(description)

# Display detailed segment statistics
print("\nDetailed segment statistics:")
print(analysis)

# Display category analysis
print("\nCategory analysis per segment:")
print(category_analysis)

# Display loyalty status analysis
print("\nLoyalty status analysis per segment:")
print(loyalty_analysis)