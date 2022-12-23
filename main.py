from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import re
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
import glob
from natsort import natsorted



def get_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    #save soup to html file named 'index.html'
    with open('index.html', 'w', encoding='utf-8-sig') as file:
        file.write(soup.prettify())
    return soup

def parse(soup):
    results = soup.find_all('div', {'class': 's-item__wrapper clearfix'})
    #<div class="s-item__title"><span role="heading" aria-level="3">2003 Dragon Ball Z Score Majin Vegeta, The Malicious Limited Foil Rare Mint 8.5</span></div>
    list_of_dicts = []
    
    for result in results:
        title = result.find('div', {'class': 's-item__title'}).text
        title = make_safe_filename(title)
        #remove spaces and replace with underscore
        title = title.replace(' ', '_')
        price = result.find('span', {'class': 's-item__price'}).text
        #<span class="s-item__shipping s-item__logisticsCost"><span class="ITALIC">+$23.50 shipping</span></span>
        shipping = result.find('div', {'class': 's-item__detail s-item__detail--primary'}).text
        #<span class="POSITIVE">Sold  Dec 21, 2022</span>
        
        if result.find('span', {'class': 'POSITIVE'}):
            sold_date = result.find('span', {'class': 'POSITIVE'}).text
        else:
            sold_date = None
        

        img_link = result.find('img', {'class': 's-item__image-img'})['src']
        item_page = result.find('a', {'class': 's-item__link'})['href']
        
        dictionary = {
            'title': title,
            'price': price,
            'shipping': shipping,
            'sold_date': sold_date,
            'img_link': img_link,
            'item_page': item_page,
        }
    
        list_of_dicts.append(dictionary)
    return list_of_dicts


def make_safe_filename(string):
    # Replace any characters that are not safe for use in a filename
    # with an underscore
    safe_string = re.sub(r'[^\w\s_.-]', '_', string)
    return safe_string


        
def download_imgs(search_term):
    #open csv file
    df = pd.read_csv(f'{search_term}.csv')
    #create a folder named 'images'
    if not os.path.exists('images'):
        os.makedirs('images')
    #download images
    for index, row in df.iterrows():
        #get image link and title for each row and save it to the images folder with the title as the file name
        img_link = row['img_link']
        img_name = row['title']
        
        img_data = requests.get(img_link).content
        print(f'Downloading {img_name}...out of {len(df)}')
        with open(f'images/{img_name}.jpg', 'wb') as handler:
            handler.write(img_data)


def scrape_to_csv(url):
    list_of_dicts = []
    soup = get_data(url)
    dict_list = parse(soup)
    list_of_dicts.extend(dict_list)
    #<a href="https://www.ebay.com/sch/i.html?_from=R40&amp;_nkw=dbz+score+graded&amp;_sacat=0&amp;rt=nc&amp;LH_Sold=1&amp;LH_Complete=1&amp;_pgn=2" _sp="p2351460.m4115.l8631" data-track="{&quot;eventFamily&quot;:&quot;LST&quot;,&quot;eventAction&quot;:&quot;ACTN&quot;,&quot;actionKind&quot;:&quot;NAVSRC&quot;,&quot;actionKinds&quot;:[&quot;NAVSRC&quot;],&quot;operationId&quot;:&quot;2351460&quot;,&quot;flushImmediately&quot;:false,&quot;eventProperty&quot;:{&quot;moduledtl&quot;:&quot;mi%3A4115%7Ciid%3A1%7Cli%3A1514%7Cluid%3Anext%7Ckind%3Apages%7C&quot;,&quot;pageci&quot;:&quot;f99d25f6-8255-11ed-8e0c-e2d019f6c0e5&quot;,&quot;parentrq&quot;:&quot;3c4f9e391850adb9c9b5b858fffbe0bc&quot;}}" type="next" class="pagination__next icon-link" aria-label="Go to next search page" style="min-width:40px;"><svg class="icon icon--pagination-next" focusable="false" aria-hidden="true"><use xlink:href="#icon-pagination-next"></use></svg></a>
        #if a next button exists, get the link and call the function again
    
    while True:
        next_button = soup.find('a', {'class': 'pagination__next icon-link'})
        if next_button:
            url = next_button['href']
            soup = get_data(url)
            dict_list = parse(soup)
            list_of_dicts.extend(dict_list)
        else:
            break
        
    #save to csv
    df = pd.DataFrame(list_of_dicts)
    df.to_csv(f'{search_term}.csv', index=False, encoding='utf-8-sig')
    
    download_imgs(search_term=search_term)
       
def insert_images():
    #convert to csv to html
    df = pd.read_csv(f'{search_term}.csv')
    
        
    df.to_html('final.html')
    
    
    


if __name__ == '__main__':
    #input url with sold items and completed listings
    search_term = 'dbz score graded'
#add plus sign between words
    search_term = search_term.replace(' ', '+')
    url = f'https://www.ebay.com/sch/i.html?_from=R40&_nkw={search_term}&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1'
    #scrape_to_csv(url)
    insert_images()