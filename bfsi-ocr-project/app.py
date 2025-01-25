import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Load data
data_file = "extracted_data.csv"
df = pd.read_csv(data_file)

# Function to generate bar plot
def generate_bar_plot(df):
    sums = df[['credit', 'debit']].sum()
    plt.figure(figsize=(10, 6))
    plt.bar(sums.index, sums.values, color=['green', 'red'])
    plt.title('Total Credit vs Debit')
    plt.ylabel('Amount')
    plt.tight_layout()
    return plt

# Function to generate pie chart
def generate_pie_chart(df):
    df['cleaned_desc'] = df['desc'].apply(lambda x: re.sub(r'\d+', '', str(x)).strip())
    debit_summary = df[df['debit'] > 0].groupby('cleaned_desc')['debit'].sum().sort_values(ascending=False)

    plt.figure(figsize=(15, 8))
    colors = plt.cm.Greens(np.linspace(0, 1, len(debit_summary)))
    wedges, texts, autotexts = plt.pie(
        debit_summary.values, 
        colors=colors, 
        autopct='%1.1f%%', 
        pctdistance=0.85, 
        startangle=180, 
        textprops={'color': 'black'}
    )
    plt.legend(wedges, debit_summary.index, title="Transaction Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("Debit Transaction Breakdown")
    plt.tight_layout()
    return plt

# Function to convert plot to PNG for download
def plot_to_png(plt):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Streamlit UI
st.title("Transaction Data Visualization")

# Show bar plot
st.header("Bar Plot: Total Credit vs Debit")
bar_plot = generate_bar_plot(df)
st.pyplot(bar_plot)
bar_plot_buffer = plot_to_png(bar_plot)
st.download_button(
    label="Download Bar Plot as PNG",
    data=bar_plot_buffer,
    file_name="bar_plot.png",
    mime="image/png"
)

# Show pie chart
st.header("Pie Chart: Debit Transaction Breakdown")
pie_chart = generate_pie_chart(df)
st.pyplot(pie_chart)
pie_chart_buffer = plot_to_png(pie_chart)
st.download_button(
    label="Download Pie Chart as PNG",
    data=pie_chart_buffer,
    file_name="pie_chart.png",
    mime="image/png"
)
