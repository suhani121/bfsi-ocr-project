# bfsi-ocr-project

This project focuses on OCR-based data extraction, data visualization, transaction analysis using BERT, and customer categorization through supervised, semi-supervised, and unsupervised learning techniques. Below is a detailed guide to using the repository.

## Prerequisites
Make sure you have Python installed (preferably version 3.8) and have `pip` set up.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Supervised Learning Workflow
1. Navigate to the `supervised` folder:
   ```bash
   cd supervised
   ```
2. Extract data using OCR:
   ```bash
   python OCR_extraction.py
   ```
   This will generate a file named `extracted_data.csv`.
3. Visualize the extracted data:
   ```bash
   python data_analysis.py
   ```
4. Train the BERT model for transaction classification:
   ```bash
   python bert_classification.py transaction.py
   ```
   Once the model is trained, it will be saved in the specified directory.
5. Analyze the classification results:
   ```bash
   python classification_analysis.py
   ```
6. Launch the web app:
   ```bash
   streamlit run app.py
   ```
   Upload files from the `supervised/Bank` folder and check the generated visuals on the app.

## Semi-Supervised Learning Workflow
The `semi-supervised` folder includes:
- **Weather API Usage**: Fetch weather data using an API. Put your API key you get from the weather site.
- **Web Scraping**: Scrape financial data from a specified website.

To use this workflow:
1. Navigate to the `semi_supervised` folder:
   ```bash
   cd semi_supervised
   ```
2. Run the geocode and weather API script:
   ```bash
   python Geocode.py
   ```
   ```bash
   python Weather.py
   ```
3. Run the web scraping script:
   ```bash
   python webscrapebs4.py
   ```

## Unsupervised Learning Workflow
The `unsupervised` folder includes a K-Means algorithm to predict categories based on customer patterns.

To use this workflow:
1. Navigate to the `unsupervised` folder:
   ```bash
   cd unsupervised
   ```
2. Run the K-Means categorization script:
   ```bash
   python kmeans_cluster.py
   ```

## Project Structure
```
├── supervised
│   ├── Bank dataset
│   ├── OCR_extraction.py
│   ├── data_analysis.py
│   ├── bert_classification.py
│   ├── transaction_dataset.csv
│   ├── classification_analysis.py
├── semi_supervised
│   ├── Geocode.py
│   ├── Weather.py
│   └── webscrapebs4.py
├── unsupervised
│   ├── kmeans_cluster.py
├── requirements.txt
├── app.py
└── README.md
```

## Notes
- Ensure all required APIs are configured before running the scripts in the `semi_supervised` folder.
- Update file paths and configurations as necessary.

## Contribution
Feel free to contribute to this project by submitting a pull request. For major changes, please open an issue to discuss what you would like to change.

## License
This project is licensed under the MIT License.

