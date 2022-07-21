import csv
import pandas as pd
#from pprint import pprint
import datetime
import re
import requests
from time import sleep
from bs4 import BeautifulSoup
from selectorlib import Extractor
import yaml
from yaml.loader import SafeLoader
import json
import numpy as np

import os
import streamlit as st
import boto3

from io import StringIO
import ast

import streamlit as st
import sys
from pathlib import Path
from src import config as cf
from src.util import data_manager as dm


class BookingScrapper():   
       
    def __init__(self):
        self.accommodation_id = None
        self.description = None
        self.nearby_places = None
        self.location_highlights = None
        self.facilities = None
        self.policies = None
        self.review_scores = None
        self.children_policies = None
        self.payment_features = None
        self.room_list = None
        self.map_markers = None
        self.reviews = None
        self.reviews_filter_metadata = None
        self.locations = None
        self.html_headers = None
        self.url = None
        self.city = None   
       
    
    '''
    def get_api_header(self):
        headers = {
            'x-rapidapi-key': "a8b6fd745amshd1ae2e4a39d3950p193801jsn3549c554af64",
            'x-rapidapi-host': "booking-com.p.rapidapi.com"
        } 
        
        return headers
    '''

    
    def get_structure(self):
        """
        file_name = os.path.join(cf.S3_DATA_BOOKING, 'booking_structure.yml')
        with open(file_name) as f:
            booking_structure = yaml.load(f, Loader=SafeLoader)
        """
        file_name = "/".join([cf.S3_DATA_BOOKING, 'booking_structure.yml'])
        response = cf.S3_CLIENT.get_object(Bucket=cf.S3_DATA_PATH, Key= file_name)
        booking_structure = yaml.load(response.get("Body"), Loader=SafeLoader)

        return booking_structure


    """
    # call to search location rapid apid to get dest_id and num of hotels of the destination
    """
    def search_location(self, dest_name='london'):
        list_by_map_querystring = {
                'name': dest_name,
                'locale': 'en-gb',
            }
        list_by_map_response = requests.request("GET", cf.BOOKING_SEARCH_LOCATION, headers=cf.BOOKING_RAPIDAPID_QUERYSTRING, params=list_by_map_querystring).json()     
        dest_id = list_by_map_response[0]['dest_id']
        nr_hotels = list_by_map_response[0]['nr_hotels']
        
        return dest_id, nr_hotels



    """
    Get list of hotels store results in json file and store a summerized main info in a csv file
    """
    def search_accommodation(self, dest_id='-2601889', filter_by_currency='GBP', city='London', nr_hotels=20):
        self.city = city

        st.markdown('#### Start downloading hotels and storing on S3 ')

        # set a checkin date, checkout date in the future to get data
        in_d = datetime.datetime.today() + datetime.timedelta(days=cf.data['default_num_of_day'])
        in_d = str(in_d.year) + '-' + str(in_d.month)  + '-' +  str(in_d.day)
        out_d = datetime.datetime.today() + datetime.timedelta(days=cf.data['default_num_of_day'] + 1)
        out_d = str(out_d.year) + '-' + str(out_d.month)  + '-' +  str(out_d.day)

        # Define search conditions through setting values for paramters in url's request
        booking_list = []
        nr_pages = int(np.round(nr_hotels/cf.data['num_of_hotels_per_page']))
        st.markdown('#### Number of pages to download: ')
        st.write('Number of pages: ',  str(nr_pages))
        st.write('Page number: ')
        for page_numer in range(0, nr_pages):
            st.write(page_numer)
            list_by_map_querystring = {
                'units': 'metric',
                'order_by': 'popularity',
                'checkin_date': in_d,
                'filter_by_currency': filter_by_currency,
                'adults_number': '2',
                'checkout_date': out_d,
                'dest_id': dest_id,
                'locale': 'en-gb',
                'dest_type': 'city',
                'room_number': '1',
                'page_number': page_numer,
            } 
            list_by_map_response = requests.request("GET", cf.BOOKING_SEARCH_HOTEL, headers=cf.BOOKING_RAPIDAPID_QUERYSTRING, params=list_by_map_querystring).json()

            # store results from rapidapi to s3
            for result in list_by_map_response['result']:
                accommodation_id = result['hotel_id']
                review_nr = result['review_nr']
                hotel_name = result['hotel_name']
                zipcode = result['zip']
                url = result['url']
                max_photo_url = result['max_photo_url']
                booking_list.append([accommodation_id,hotel_name,review_nr,zipcode,url,max_photo_url])
                file_name =  str(accommodation_id) + '.json'
                dm.write_json_file(bucket_name=cf.S3_DATA_PATH, 
                                   file_name= "/".join([cf.S3_DATA_BOOKING, city,cf.S3_BOOKING_HOTEL, file_name]),
                                   data=result, type='s3')
         
        # create a list to store main info for next steps
        booking_list = pd.DataFrame(booking_list)
        booking_list.columns = ['hotel_id','hotel_name','review_nr','zipcode','url','max_photo_url']
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, cf.HOTEL_LIST_FILE]), 
                          data=booking_list, type='s3')

        st.markdown('#### Finished downloading hotels and storing on S3')
        st.write(booking_list.head(10))
        

    
    """
    Get reviews of a hotel and store them json file
    """    
    def get_review(self):

        st.markdown('#### Start downloading reviews and storing on S3: ')
        hotel_list = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH, file_name="/".join([cf.S3_DATA_BOOKING, self.city,cf.HOTEL_LIST_FILE]), type='s3')
        st.markdown('#### Number of hotels: ')
        st.write(len(hotel_list))
        st.markdown('#### Hotel number: ')
        hotel_num = 0
        for accommodation_id in hotel_list.hotel_id:
            hotel_num = hotel_num + 1
            st.write(hotel_num)
            num_of_page = int(cf.data['review_num_of_page'])
            for page_number in range(0, num_of_page):
                querystring = {"locale":"en-gb","hotel_id": accommodation_id}
                review_querystring = {"sort_type":"SORT_MOST_RELEVANT","locale":"en-gb",
                              "hotel_id":accommodation_id,"language_filter":"en-gb",
                              "page_number": page_number}
                headers = cf.BOOKING_RAPIDAPID_QUERYSTRING               
                structure = self.get_structure()
                try:
                    data = {}
                    data['accommodation_id'] = accommodation_id
                    #data['nearby_places'] = requests.request("GET", structure['nearby_places'], headers=headers, params=querystring).text    
                    data['reviews'] = requests.request("GET", structure['reviews'], headers=headers, params=review_querystring).text
                    # store data to s3
                    file_name = 'hotel_' + str(accommodation_id) + '_' + str(page_number) + '.json'
                    dm.write_json_file(bucket_name=cf.S3_DATA_PATH, 
                                   file_name="/".join([cf.S3_DATA_BOOKING, self.city,cf.S3_BOOKING_REVIEW, file_name]), 
                                   data=data, type='s3')
                except:
                    st.write('error')

        st.markdown('#### Finished downloading reviews and storing on S3')


    """
    Collect all reviews into one file
    """   
    def create_review_file(self, city):
        self.city = city
        bucket_name = cf.S3_DATA_PATH
        my_bucket = cf.S3_RESOURCE.Bucket(bucket_name)
        accommodation_review = []

        st.markdown("#### Reading files from S3. Please wait...")
        #for file in all_objects['Contents']:
        for file in my_bucket.objects.filter(Prefix="/".join([cf.S3_DATA_BOOKING, city, cf.S3_BOOKING_REVIEW, ''])):
            filename = file.key
            response = cf.S3_CLIENT.get_object(Bucket=bucket_name, Key=filename)
            reviews_data = json.loads(response.get("Body").read())
            accommodation_id = reviews_data['accommodation_id'] 
            reviews = reviews_data['reviews']
            try:
                reviews = ast.literal_eval(reviews)
                review_result = reviews['result']
                for i in range(0, len(review_result)):
                    review = review_result[i]
                    pros = review['pros']
                    cons = review['cons']
                    review = pros + '. ' + cons
                    accommodation_review.append([accommodation_id, pros, cons, review])
            except:
                pass
        accommodation_review_df = pd.DataFrame(accommodation_review, columns = ['accommodation_id', 'pros', 'cons', 'review'])
        dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                          file_name="/".join([cf.S3_DATA_BOOKING, city, 'review.csv']), 
                          data=accommodation_review_df, type='s3')

        st.markdown('#### Finished reading reviews from S3 ')
        st.markdown('#### Number of reviews: ')
        st.write(accommodation_review_df.shape[0])
        st.markdown('#### Show samples of reviews: ')
        st.write(accommodation_review_df.head(20))


    def scrape_html(self, url):
        
        # Create an Extractor by reading from the YAML file
        e = Extractor.from_yaml_file('booking.yml')
        
        headers = {
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1',
            # You may want to change the user agent if you get blocked
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Referer': 'https://www.booking.com/index.en-gb.html',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        }
        
        # Download the page using requests
        print("Downloading %s"%url)
        r = requests.get(url, headers=headers)
        # Pass the HTML of the page and create
        return e.extract(r.text, base_url=url)


    def get_search_result(self):
        with open("urls.txt",'r') as urllist, open('booking_list.csv','w') as outfile:
            fieldnames = [
                "name",
                "location",
                "price",
                "price_for",
                "room_type",
                "beds",
                "rating",
                "rating_title",
                "number_of_ratings",
                "url"
            ]        
            writer = csv.DictWriter(outfile, fieldnames=fieldnames,quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for url in urllist.readlines():
                data = self.scrape_html(url) 
                if data:
                    print(data)
                    for h in data['hotels']:
                        writer.writerow(h) 

 