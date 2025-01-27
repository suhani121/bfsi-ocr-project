import os
import pytesseract
from PIL import Image
import pandas as pd
import re
import numpy as np
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import pdfplumber

# Set up Streamlit page
st.set_page_config(page_title="BFSI-OCR", layout="wide")

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

        # Bar Chart: Credit vs Debit
        sums = result_df[['credit', 'debit']].sum()
        fig, ax = plt.subplots()
        ax.bar(sums.index, sums.values, color=['green', 'red'])
        ax.set_title('Total Credit vs Debit')
        ax.set_ylabel('Amount')
        st.pyplot(fig)

        # Pie Chart: Debit Transaction Breakdown
        st.subheader("Debit Transaction Breakdown")
        debit_transactions = result_df[result_df['debit'] > 0]
        debit_summary = debit_transactions.groupby('cleaned_description')['debit'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(15, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(debit_summary)))
        wedges, texts, autotexts = ax.pie(
            debit_summary.values,
            colors=colors,
            autopct='%1.1f%%',
            pctdistance=0.85,
            startangle=180,
            textprops={'color': 'black'}
        )
        ax.legend(
            wedges,
            debit_summary.index,
            title="Transaction Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        ax.set_title("Debit Transaction Breakdown")
        plt.tight_layout()
        st.pyplot(fig)

        # Additional Pie Chart: Distribution of Predicted Categories
        st.subheader("Distribution of Predicted Categories")
        try:
            output_file = "unsupervised/classified_data.csv"
            classified_df = pd.read_csv(output_file)

            category_counts = classified_df['predicted_category'].value_counts()

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                category_counts, 
                labels=category_counts.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=plt.cm.tab20.colors
            )
            ax.set_title('Distribution of Predicted Categories')
            plt.axis('equal')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error processing classified data pie chart: {str(e)}")

    else:
        st.warning("No valid data was extracted.")
