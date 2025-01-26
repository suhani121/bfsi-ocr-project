import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('synthetic_transactions.csv')

# Check the structure 
print(df.head())  
amounts = df[['Amount']]

# Standardize the data
scaler = StandardScaler()
amounts_scaled = scaler.fit_transform(amounts)

# Apply K-means clustering 
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(amounts_scaled)

# Display the results
print(df[['Description', 'Amount', 'Cluster']])

# Optional: Plot the clusters
plt.figure(figsize=(10,6))  # Adjust the figure size 
plt.scatter(df['Description'], df['Amount'], c=df['Cluster'], cmap='viridis')
plt.xticks(rotation=70)  
plt.xlabel('Description')
plt.ylabel('Amount')
plt.title('Transaction Clusters by Amount')

plt.subplots_adjust(top=0.9)  

plt.show()
