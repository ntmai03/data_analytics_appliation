import csv
import pandas as pd
from pprint import pprint
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
sys.path.append('src')

sys.path.append('src')
from src import config as cf


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
        self.city_path = None
        self.hotel_path = None
        self.review_path = None



       
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
    
    
    def get_api_header(self):

        '''
        with open('booking_config.yml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            headers = {
                #'x-rapidapi-key': config['booking']['key'],
                #'x-rapidapi-host': config['booking']['host']
                'x-rapidapi-key': "a8b6fd745amshd1ae2e4a39d3950p193801jsn3549c554af64",
                'x-rapidapi-host': "booking-com.p.rapidapi.com"

            }   
        '''  

        headers = {
            'x-rapidapi-key': "a8b6fd745amshd1ae2e4a39d3950p193801jsn3549c554af64",
            'x-rapidapi-host': "booking-com.p.rapidapi.com"

        } 
        
        return headers
    
    
    def get_structure(self):
        """
        file_name = os.path.join(cf.S3_DATA_BOOKING, 'booking_structure.yml')
        with open(file_name) as f:
            booking_structure = yaml.load(f, Loader=SafeLoader)
        """
        response = cf.S3_CLIENT.get_object(Bucket=cf.S3_DATA_PATH, Key=cf.S3_DATA_BOOKING + 'booking_structure.yml')
        booking_structure = yaml.load(response.get("Body"), Loader=SafeLoader)

        return booking_structure

    
    """
    This method reads in tweet data as Json and extracts the data we want
    """
    def get_data_from_API(self, accommodation_id, page):

        querystring = {"locale":"en-gb","hotel_id": accommodation_id}
        review_querystring = {"sort_type":"SORT_MOST_RELEVANT","locale":"en-gb",
                              "hotel_id":accommodation_id,"language_filter":"en-gb",
                              "page_number": page}
        headers = self.get_api_header()                
        structure = self.get_structure()

        try:
            self.accommodation_id = accommodation_id
            #self.description = requests.request("GET", structure['description'], headers=headers, params=querystring).text
            self.nearby_places = requests.request("GET", structure['nearby_places'], headers=headers, params=querystring).text    
            self.location_highlights = requests.request("GET", structure['location_highlights'], headers=headers, params=querystring).text
            self.reviews = requests.request("GET", structure['reviews'], headers=headers, params=review_querystring).text
        
        except Error as e:
            print(e)
            
    
    def save_to_json(self, page):
        
        structure = self.get_structure()  
        data = {}
        
        
        data['accommodation_id'] = self.accommodation_id
        #data['description'] = self.description
        data['nearby_places'] = self.nearby_places
        data['location_highlights'] = self.location_highlights
        #data['facilities'] = self.facilities
        #data['policies'] = self.policies
        #data['review_scores'] = self.review_scores
        #data['children_policies'] = self.children_policies
        #data['payment_features'] = self.payment_features
        #data['room_list'] = self.room_list
        #data['map_markers'] = self.map_markers
        data['reviews'] = self.reviews
        #data['reviews_filter_metadata'] = self.reviews_filter_metadata
        #data['locations'] = self.locations
            
        """
        file_name = self.review_path + '/hotel_' + str(self.accommodation_id) + '.json'
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile)
        """
        file_name = 'hotel_' + str(self.accommodation_id) + '_' + str(page) + '.json'
        cf.S3_CLIENT.put_object(
            Bucket=self.city + '-booking-review',
            Key=file_name,
            Body = json.dumps(data).encode('UTF-8')
        )
        '''
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=self.review_path + file_name,
            Body = json.dumps(data).encode('UTF-8')
        )
        '''

    def search_accommodation(self, dest_id='-2601889', filter_by_currency='GBP', city='London'):
        
        self.city = city
        self.city_path = os.path.join(cf.S3_DATA_BOOKING, self.city)
        self.hotel_path = self.city_path + '/hotel/'
        self.review_path = self.city_path + '/review/'

        """
        os.makedirs(self.city_path, exist_ok=True)
        os.makedirs(self.hotel_path, exist_ok=True)
        os.makedirs(self.review_path, exist_ok=True)
        """

        # Creating a bucket in AWS S3
        #cf.S3_CLIENT.put_object(Bucket=cf.S3_DATA_PATH, Key=self.city_path)
        #cf.S3_CLIENT.put_object(Bucket=cf.S3_DATA_PATH, Key=self.hotel_path)
        #cf.S3_CLIENT.put_object(Bucket=cf.S3_DATA_PATH, Key=self.review_path)


        booking_list = []
        list_by_map_url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
  

        list_by_map_headers = {
            'x-rapidapi-key': "a8b6fd745amshd1ae2e4a39d3950p193801jsn3549c554af64",
            'x-rapidapi-host': "booking-com.p.rapidapi.com"
        }

        # source code
         
        booking_list = []
        for page_numer in range(0, 1):
            st.write(page_numer)
            list_by_map_querystring = {
                'units': 'metric',
                'order_by': 'popularity',
                'checkin_date': '2021-12-25',
                'filter_by_currency': filter_by_currency,
                'adults_number': '2',
                'checkout_date': '2021-12-26',
                'dest_id': dest_id,
                'locale': 'en-gb',
                'dest_type': 'city',
                'room_number': '1',
                #'children_ages': '5,0',
                'page_number': page_numer,
                #'categories_filter_ids': 'facility::107,free_cancellation::1',
                #'children_number': '2'
            } 
            list_by_map_response = requests.request("GET", list_by_map_url, headers=list_by_map_headers, params=list_by_map_querystring).json()

            
            for list_result in list_by_map_response['result']:
                accommodation_id = list_result['hotel_id']
                review_nr = list_result['review_nr']
                hotel_name = list_result['hotel_name']
                zipcode = list_result['zip']
                url = list_result['url']
                max_photo_url = list_result['max_photo_url']

                booking_list.append([accommodation_id,hotel_name,review_nr,zipcode,url,max_photo_url])

                file_name =  str(accommodation_id) + '.json'
                '''
                with open(self.hotel_path + '/' + file_name, 'w') as outfile:
                    json.dump(list_result, outfile)
                '''

                # store in S3
                #s3 = boto3.resource('s3')
                #s3object = s3.Object('your-bucket-name', 'your_file.json')
                cf.S3_CLIENT.put_object(
                    Bucket=cf.S3_DATA_PATH,
                    Key=self.hotel_path + file_name,
                    Body = json.dumps(list_result).encode('UTF-8')

                )
         
        booking_list = pd.DataFrame(booking_list)
        booking_list.columns = ['hotel_id','hotel_name','review_nr','zipcode','url','max_photo_url']
        # booking_list.to_csv(self.city_path + '/booking_list.csv', index=False)
        csv_buffer = StringIO()
        booking_list.to_csv(csv_buffer)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=self.city_path + '/booking_list.csv',
            Body = csv_buffer.getvalue()

        )
        
        
    def get_review(self):
        
        # hotel_list = pd.read_csv(self.city_path + '/booking_list.csv')
        response = cf.S3_CLIENT.get_object( Bucket=cf.S3_DATA_PATH, Key=self.city_path + '/booking_list.csv')
        hotel_list = pd.read_csv(response.get("Body"))
        for hotel_id in hotel_list.hotel_id:
            try:
                #num_of_page = hotel_list[hotel_list.hotel_id == hotel_id].review_nr[0] / 25.0
                #num_of_page = num_of_page.astype(int)
                num_of_page = 5
            except:
                pass

            for page_numer in range(0, num_of_page):
                st.write(hotel_id, page_numer)
                self.get_data_from_API(hotel_id, page_numer)
                self.save_to_json(page_numer)


    def create_review_file(self, city):

        self.city = city
        self.city_path = os.path.join(cf.S3_DATA_BOOKING, self.city)
        self.hotel_path = self.city_path + '/hotel/'

        Bucket = self.city + '-booking-review'
        all_objects = cf.S3_CLIENT.list_objects(Bucket=Bucket) 
        accommodation_review = []

        for file in all_objects['Contents']:
            filename = file['Key']
            response = cf.S3_CLIENT.get_object(Bucket=Bucket, Key=filename)
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
        review_buffer = StringIO()
        accommodation_review_df.to_csv(review_buffer)
        cf.S3_CLIENT.put_object(
            Bucket=cf.S3_DATA_PATH,
            Key=self.city_path + '/review.csv',
            Body = review_buffer.getvalue()

        )
        st.write('done')
        st.write(self.city_path + '/review.csv')

 