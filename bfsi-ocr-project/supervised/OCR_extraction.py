import os
import pytesseract
from PIL import Image
import pandas as pd
import re
import cv2
import pdfplumber

# Define paths
root_folder = r"Bank"
subfolders = ["Bank Statements", "Invoices", "Pay Slips"]
output_csv = "extracted_data.csv"

# Function to extract numbers near keywords
def extract_numbers_near_keywords(text, keywords):
    for keyword in keywords:
        pattern = rf"{keyword}[:\s]*([\d,.\s]+)" # Check if its at least 3 digits
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            extracted_number = match.group(1).replace(",", "").strip()

            # Validate the extracted number
            try:
                extracted_number = float(extracted_number)
                if extracted_number >= 100:  
                    keyword_position = match.start()
                    keyword_line_start = text.rfind("\n", 0, keyword_position)
                    keyword_line_end = text.find("\n", keyword_position)
                    keyword_line = text[keyword_line_start + 1:keyword_line_end].strip()

                    next_line_start = text.find("\n", keyword_line_end) + 1
                    next_line_end = text.find("\n", next_line_start)
                    next_line = text[next_line_start:next_line_end].strip()

                    if str(int(extracted_number)) in keyword_line or str(int(extracted_number)) in next_line:
                        return extracted_number
            except ValueError:
                continue  

    return 0.0

# Finding table
def extract_largest_table(text):
    tables = text.split("\n\n") 
    max_table = max(tables, key=lambda table: len(table.split("\n")), default="")
    return max_table.strip()

# Cropping
def crop_transaction_area(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    cropped = image[int(height * 0.3):int(height * 0.9), int(width * 0.1):int(width * 0.9)]  # Example: center-middle area
    return cropped

# Function to process bank statements from PDF
def process_bank_statement(pdf_path):
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        transactions = []
        
        # Process each page
        for page in pdf.pages:
            # Extract table from the page
            table = page.extract_table()
            
            if table:
                # Process each row in the table
                for row in table:
                    # Skip header rows and empty rows
                    if row and 'Txn Date' not in str(row[0]) and row[0] is not None and row[0].strip().isdigit():
                        try:
                            # Extract relevant columns
                            # Assuming columns: Srl, Txn Date, Value Date, Description, Cheque No, CR/DR, CCY, Amount, Balance
                            txn_date = row[1]
                            description = str(row[3]) if row[3] else ''
                            cr_dr = str(row[5]) if row[5] else ''
                            amount_str = str(row[7]) if row[7] else '0'
                            
                            # Clean amount string
                            amount = float(amount_str.replace('INR', '').replace(',', '').strip())
                            
                            # Determine if credit or debit based on CR/DR column
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
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"Error processing row: {row}")
                            continue

    # Create DataFrame
    if transactions:
        df = pd.DataFrame(transactions)
        
        # Calculate totals
        total_credits = df['credit'].sum()
        total_debits = df['debit'].sum()
        
        return df, total_credits, total_debits
    else:
        raise ValueError("No transactions found in the PDF")

data = []

# Process each subfolder
for subfolder in subfolders:
    folder_path = os.path.join(root_folder, subfolder)
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        continue

    # Iterate over files in the subfolder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')):  # Support for PDFs
            continue 
        
        if subfolder == "Bank Statements" and file_path.lower().endswith('.pdf'):  # Handle PDFs in Bank Statements
            try:
                df, total_credits, total_debits = process_bank_statement(file_path)
                # Save transactions to the data list
                for index, row in df.iterrows():
                    data.append({
                        "date": row['date'],
                        "desc": row['description'],
                        "credit": row['credit'],
                        "debit": row['debit'],
                        "balance": row['balance']
                    })
                print(f"Processed {filename}: Total Credits: INR {total_credits:,.2f}, Total Debits: INR {total_debits:,.2f}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        elif subfolder == "Bank Statements":  # Process non-PDF bank statement images
            cropped_image = crop_transaction_area(file_path)
            
            temp_path = "temp_cropped_image.jpg"
            cv2.imwrite(temp_path, cropped_image)

            text = pytesseract.image_to_string(Image.open(temp_path))

            lines = text.split("\n")
            for line in lines:
                if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', line):  
                    match = re.match(r"(.*?)([\d,.]+)\s+([\d,.]+)$", line)
                    if match:
                        desc, debit, credit = match.groups()
                        data.append({
                            "desc": desc.strip(),
                            "debit": float(debit.replace(",", "")),
                            "credit": float(credit.replace(",", ""))
                        })

        elif subfolder == "Invoices":
            text = pytesseract.image_to_string(Image.open(file_path))
            
            # Extract the total or subtotal
            amount = extract_numbers_near_keywords(text, ["Total", "Subtotal"])
            data.append({
                "desc": "Export",
                "debit": amount,
                "credit": 0.0
            })
        
        elif subfolder == "Pay Slips":
            text = pytesseract.image_to_string(Image.open(file_path))
            
            # Extract the salary amount
            amount = extract_numbers_near_keywords(text, ["Net Pay", "Subtotal", "Total", "Grand Total"])
            data.append({
                "desc": "Salary Payment",
                "debit": 0.0,
                "credit": amount
            })

# Save
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Extraction complete. Data saved.")
