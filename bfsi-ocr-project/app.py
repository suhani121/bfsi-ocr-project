import os
import pytesseract
from PIL import Image
import pandas as pd
import re
import numpy as np
import streamlit as st
from io import BytesIO
import pdfplumber
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px  # Importing Plotly for interactive plots

# Set up Streamlit page
st.set_page_config(page_title="BFSI-OCR", layout="wide")

# Sidebar with options
st.sidebar.title("Select the type of analysis")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Supervised", "Unsupervised", "Semi-Supervised"])

# Function to extract numbers near keywords
def extract_numbers_near_keywords(text, keywords):
    for keyword in keywords:
        pattern = rf"{keyword}[:\s]*([\d,.\s]+)"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            extracted_number = match.group(1).replace(",", "").strip()
            try:
                extracted_number = float(extracted_number)
                if extracted_number >= 100:
                    return extracted_number
            except ValueError:
                continue
    return 0.0

# Function to process bank statements from PDF
def process_bank_statement(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        transactions = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if row and 'Txn Date' not in str(row[0]) and row[0] is not None and row[0].strip().isdigit():
                        try:
                            txn_date = row[1]
                            description = str(row[3]) if row[3] else ''
                            cr_dr = str(row[5]) if row[5] else ''
                            amount_str = str(row[7]) if row[7] else '0'
                            amount = float(amount_str.replace('INR', '').replace(',', '').strip())
                            if 'Cr.' in cr_dr:
                                transactions.append({
                                    'date': txn_date,
                                    'description': description,
                                    'type': 'Credit',
                                    'credit': amount,
                                    'debit': 0,
                                    'balance': row[8].replace('INR', '').replace(',', '').strip()
                                })
                            elif 'Dr.' in cr_dr:
                                transactions.append({
                                    'date': txn_date,
                                    'description': description,
                                    'type': 'Debit',
                                    'credit': 0,
                                    'debit': amount,
                                    'balance': row[8].replace('INR', '').replace(',', '').strip()
                                })
                        except (ValueError, TypeError, IndexError):
                            continue
    return pd.DataFrame(transactions)

# Supervised Section
if analysis_type == "Supervised":
    # Streamlit File Uploader
    st.title("BFSI-OCR")
    st.subheader("Upload your Bank Statements, Invoices, or Pay Slips")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        data = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()

            # Process PDF files
            if file_name.endswith(".pdf"):
                try:
                    st.write(f"Processing PDF: {uploaded_file.name}")
                    df = process_bank_statement(uploaded_file)
                    data.extend(df.to_dict('records'))
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            # Process Image files
            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                try:
                    st.write(f"Processing Image: {uploaded_file.name}")
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image)
                    amount = extract_numbers_near_keywords(text, ["Total", "Net Pay", "Subtotal"])

                    # Add extracted image data
                    data.append({
                        "description": "Extracted from image OCR",
                        "credit": amount if amount > 0 else 0.0,
                        "debit": 0.0 if amount > 0 else abs(amount),
                        "balance": 0.0  # Default balance for image data
                    })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # After collecting all the data, process it into a DataFrame
        if data:
            result_df = pd.DataFrame(data)

            # Ensure the 'description' column exists
            if 'description' not in result_df.columns:
                result_df['description'] = ''  # Add a blank description column if missing

            # Clean and process the 'description' column
            def clean_description(description):
                if isinstance(description, str):
                    return re.sub(r'\d+', '', description).strip()
                return ''  # Return empty string if description is not a string

            result_df['cleaned_description'] = result_df['description'].apply(clean_description)

            # Display results
            st.subheader("Extracted Data")
            st.dataframe(result_df)

            # Save and download CSV
            csv_buffer = BytesIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="Download Extracted Data as CSV",
                data=csv_buffer,
                file_name="extracted_data.csv",
                mime="text/csv"
            )

            # Data Visualization
            st.subheader("Data Visualization")

            # Dropdown menu for chart selection
            chart_option = st.selectbox(
                "Select a chart to display:",
                ["Total Credit vs Debit", "Debit Transaction Breakdown", "Distribution of Predicted Categories"]
            )

            # Bar Chart: Credit vs Debit
            if chart_option == "Total Credit vs Debit":
                sums = result_df[['credit', 'debit']].sum()

                # Create interactive Plotly bar chart
                fig = px.bar(
                    x=sums.index, y=sums.values, labels={'x': 'Type', 'y': 'Amount'},
                    title='Total Credit vs Debit',
                    hover_data={'x': sums.index, 'y': sums.values},
                    color=sums.index, color_discrete_map={'credit': 'green', 'debit': 'red'}
                )
                st.plotly_chart(fig)

                # Add summary text
                st.markdown("**Summary**: This chart shows the total Credit and Debit amounts. The Credit values are shown in green, and the Debit values are shown in red. It helps to understand the overall flow of funds.")

            # Pie Chart: Debit Transaction Breakdown
            elif chart_option == "Debit Transaction Breakdown":
                debit_transactions = result_df[result_df['debit'] > 0]
                debit_summary = debit_transactions.groupby('cleaned_description')['debit'].sum().sort_values(ascending=False)

                # Create interactive Plotly pie chart
                fig = px.pie(
                    names=debit_summary.index, values=debit_summary.values, title="Debit Transaction Breakdown",
                    hover_data={'names': debit_summary.index, 'values': debit_summary.values}
                )
                st.plotly_chart(fig)

                # Add summary text
                st.markdown("**Summary**: This pie chart breaks down the Debit transactions by their descriptions. It shows the proportion of each category of Debit transaction.")

            # Additional Pie Chart: Distribution of Predicted Categories
            elif chart_option == "Distribution of Predicted Categories":
                try:
                    output_file = "unsupervised/classified_data.csv"
                    classified_df = pd.read_csv(output_file)

                    category_counts = classified_df['predicted_category'].value_counts()

                    # Create interactive Plotly pie chart
                    fig = px.pie(
                        names=category_counts.index, values=category_counts.values,
                        title="Distribution of Predicted Categories", hover_data={'names': category_counts.index, 'values': category_counts.values}
                    )
                    st.plotly_chart(fig)

                    # Add summary text
                    st.markdown("**Summary**: This chart shows the distribution of predicted categories. The slices represent different categories, and the size of each slice represents the proportion of each category.")

                except Exception as e:
                    st.warning(f"Error processing classified data pie chart: {str(e)}")

        else:
            st.warning("No valid data was extracted.")

# Unsupervised and Semi-Supervised sections (left empty for now)
elif analysis_type == "Unsupervised":
    st.title("Unsupervised Clustering")
    st.subheader("Upload a CSV file to apply K-Means Clustering on Transaction Data")

    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Check if the required columns are present
        if "Date" in df.columns and "Amount" in df.columns:
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

            # Create an interactive scatter plot with hover functionality
            fig = px.scatter(
                df, x="Days", y="Amount", color="Cluster", hover_data=["Date", "Amount", "Cluster"],
                title="Transaction Clustering Based on Amount & Days",
                labels={'Days': 'Days Since First Transaction', 'Amount': 'Transaction Amount'}
            )
            st.plotly_chart(fig)

            # Add summary text
            st.markdown("**Summary**: This scatter plot visualizes transactions based on the number of days since the first transaction and their amounts. Different clusters are represented by different colors, helping to identify patterns in the transaction data.")

        else:
            st.error("CSV file must contain 'Date' and 'Amount' columns.")

elif analysis_type == "Semi-Supervised":
    st.title("Semi-Supervised Analysis")
    st.subheader("Upload a CSV file with 'Company' and 'Price' data")

    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Check if the required columns are present
        if "Company" in df.columns and "Price" in df.columns:
            # Clean Price column by removing '$' and converting to float
            df["Price"] = df["Price"].replace({'\$': '', ',': ''}, regex=True).astype(float)

            # Create an interactive bar plot
            fig = px.bar(
                df, x="Company", y="Price", title="Company vs Price",
                labels={'Company': 'Company', 'Price': 'Price (USD)'},
                hover_data=["Company", "Price"]
            )
            st.plotly_chart(fig)

            # Add summary text
            st.markdown("**Summary**: This bar chart compares the prices of different companies.")

        else:
            st.error("CSV file must contain 'Company' and 'Price' columns.")
