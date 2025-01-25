import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {'user-agent':  'Chrome/84.0.4147.105'}#User agent

urls = [
    'https://groww.in/us-stocks/nke',
    'https://groww.in/us-stocks/ko', 
    'https://groww.in/us-stocks/msft', 
    'https://groww.in/stocks/m-india-ltd', 
    'https://groww.in/us-stocks/axp', 
    'https://groww.in/us-stocks/amgn', 
    'https://groww.in/us-stocks/aapl', 
    'https://groww.in/us-stocks/ba', 
    'https://groww.in/us-stocks/csco', 
    'https://groww.in/us-stocks/gs', 
    'https://groww.in/us-stocks/ibm', 
    'https://groww.in/us-stocks/intc', 
    'https://groww.in/us-stocks/jpm', 
    'https://groww.in/us-stocks/mcd',
    'https://groww.in/us-stocks/crm', 
    'https://groww.in/us-stocks/vz', 
    'https://groww.in/us-stocks/v', 
    'https://groww.in/us-stocks/wmt',  
    'https://groww.in/us-stocks/dis'
    ]


all = []
for url in urls:
    page = requests.get(url, headers=headers)
    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        company = soup.find('h1', {'class': 'usph14Head displaySmall'}).text #inspect the html elements on the site
        price = soup.find('span', {'class': 'uht141Pri contentPrimary displayBase'}).text
        x = [company, price]
        all.append(x)
    except AttributeError:
        print("Change the Element id")
    except Exception as e:
        print(f"Error occurred for URL {url}: {e}")
    time.sleep(10)

column_names = ["Company", "Price"]
df = pd.DataFrame(all, columns=column_names)

df.to_csv('stocks.csv', index=False)

print("Data extracted and saved to CSV.")
