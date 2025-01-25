import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('extracted_data.csv')

# First bar plot
sums = df[['credit', 'debit']].sum()
plt.figure(figsize=(10, 6))
plt.bar(sums.index, sums.values, color=['green', 'red'])
plt.title('Total Credit vs Debit')
plt.ylabel('Amount')
plt.show()

# Function to remove numbers and keep only alphabetic characters
def clean_description(description):
    return re.sub(r'\d+', '', description).strip()

# Clean descriptions and prepare data for pie chart
df['cleaned_desc'] = df['desc'].apply(clean_description)
debit_transactions = df[df['debit'] > 0]
debit_summary = debit_transactions.groupby('cleaned_desc')['debit'].sum().sort_values(ascending=False)

# Create pie chart with legend
plt.figure(figsize=(15, 8))
colors = plt.cm.Greens(np.linspace(0, 1, len(debit_summary)))

# Create pie chart without labels
wedges, texts, autotexts = plt.pie(debit_summary.values, 
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  pctdistance=0.85,
                                  startangle=180,
                                  textprops={'color':'#CD7F32'})

# Add a legend
plt.legend(wedges, debit_summary.index,
          title="Transaction Categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title("Debit Transaction Breakdown")

# Adjust layout to prevent legend clipping
plt.tight_layout()
plt.show()