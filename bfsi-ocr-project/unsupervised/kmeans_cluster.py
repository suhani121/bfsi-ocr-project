import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset 
df = pd.read_csv(r"bfsi-ocr-project\bfsi-ocr-project\unsupervised\synthetic_transactions.csv")

# Convert "Date" to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Compute "Days" since the first transaction
df["Days"] = (df["Date"] - df["Date"].min()).dt.days

# Standardize the "Amount" column for clustering
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Amount_Scaled"]])

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Days"], y=df["Amount"], hue=df["Cluster"], palette="Set1", s=100, edgecolor="black")

# Mark cluster centers
centers = kmeans.cluster_centers_
scaled_centers = scaler.inverse_transform(centers)  # Convert back to original scale

for i, center in enumerate(scaled_centers):
    plt.scatter(df["Days"].mean(), center, marker="X", s=300, c="black", edgecolor="white", label=f'Cluster {i} Center')

# Labels and title
plt.xlabel("Days Since First Transaction")
plt.ylabel("Transaction Amount")
plt.title("Transaction Clustering Based on Amount & Days")
plt.legend(title="Clusters")
plt.grid(True)

# Show the plot
plt.show()
