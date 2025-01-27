import pandas as pd

output_file = "classified_data.csv"
df = pd.read_csv(output_file)

category_counts = df['predicted_category'].value_counts()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.pie(
    category_counts, 
    labels=category_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.tab20.colors
)
plt.title('Distribution of Predicted Categories')
plt.axis('equal')  
plt.show()
