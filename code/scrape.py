#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# In[35]:

url = 'https://nextdoor.com/'

# set driver
PATH = '../chromedriver'
driver = webdriver.Chrome(PATH)
    # open nextdoor.com
# driver.get('https://nextdoor.com')


def nxtdr_logout():
    driver.get('https://nextdoor.com/logout/?ucl=1')


# In[23]:


def nxtdr_login(usrnm, pwd):
    
    

    time.sleep(2)
    log_in = driver.find_element_by_class_name("css-1d8yfou")
        # click log in button 
    log_in.click()

        # enter email 
    email = driver.find_element_by_id('id_email')
    email.send_keys(usrnm)
    email.send_keys(Keys.TAB)

        # enter pwd
    enter_pwd = driver.find_element_by_id("id_password")
    enter_pwd.send_keys(pwd)
        # log in 
    enter_pwd.send_keys(Keys.RETURN)
    


# In[24]:



def query(keywords):
    url = 'https://nextdoor.com/'
    driver.get(f'{url}search/?query={keywords}')
    
  


# In[25]:


def load_more(reps):
# click the load more button at the bottom of the page # of times in range(reps)'''
   
    for i in range(reps):
        try:
            show_more = driver.find_element_by_class_name('css-1on4yel')
            show_more.click()
            # allowing time for page to load
            time.sleep(3)
        except:
            
            show_more = driver.find_element_by_class_name('content-results-list-item-see-more-link')
            show_more.click()
            # allowing time for page to load
            time.sleep(3)
            
        else:
            continue


# In[26]:


def get_links():
    
    # all links on webpage ( web element )
    feed = driver.find_elements_by_tag_name('a')

    links = []

        # the first five links are advertisements
    for post in feed[5:]:
        link = post.get_attribute('href')
        links.append(link)
    # returns list of links from full loaded webpage resulting 
    # from keyword search.  
    return links

   


# In[44]:


def search( keywords, reps ):
    query(keywords)
    time.sleep(10)
    load_more(reps)
    links = get_links()
    return links
get_ipython().run_line_magic('time', '')


# In[28]:


def scrape(usrnm, pwd, keywords):
    nxtdr_login(usrnm,pwd)
    time.sleep(3)
    links = search(keywords,100)
    time.sleep(2)
    
    links_df = pd.DataFrame()
     # fill out href column with contents of links list 
    links_df['href'] = [ link for link in links]
     # clean href 
    links_df['href'] = [ href[0] for href in links_df['href'].str.split('&')]
    # drop duplicates posts
    links_df.drop_duplicates( inplace = True,   keep='first')
    # only take user posts not ads etc.
    usr_post = links_df['href'].str.contains(' ?post=')
    links_df = links_df[usr_post]
    return links_df


# In[29]:


def get_post(href):
    # open post in browser
    driver.get(href)
           # give time to load
    time.sleep(3)
        # container for post and comments 
    post_container = driver.find_element_by_class_name('css-1dkvlfs')
        # container for post
    main_post_container = post_container.find_element_by_class_name('cee-media-body')
        # actual text of main post 
    main_post = main_post_container.find_element_by_class_name('Linkify').text
        # post id 
    post_id = href.split('=')[1]

    # meta info
        # location
    main_post_location = main_post_container.find_element_by_tag_name('button').text
        # author name
    meta = main_post_container.find_elements_by_tag_name('a')
    main_post_author = meta[0].text

        # post date 
        # list of entities that mess up date pull 
    date_fix_list = ['City of Denver','News','Denver Police Department']

    if main_post_author in date_fix_list:
        main_post_date = meta[2].text
    else:
        main_post_date = meta[1].text

        # post to append to post_df
    post = {'post_id': post_id ,
           'author' : main_post_author, 'date': main_post_date,
           'location': main_post_location, 'post': main_post}
    
    return post


# In[30]:


def get_comments(href):
        # container for post and comments 
    post_container = driver.find_element_by_class_name('css-1dkvlfs')
        # container for post
    main_post_container = post_container.find_element_by_class_name('cee-media-body')
        # the boxes around each of the comments on the post
    comment_windows = post_container.find_elements_by_class_name('js-media-comment')
        # post id 
    
    
        # creating a list of comments
    comments = []
    for i in range(len(comment_windows)):
        comment = comment_windows[i].find_element_by_class_name('css-1srqc6z').text
        comment = comment.split('\n')
        comments.append(comment[1])
        
        # creating a list of locations
    locations = []
    for i in range(len(comment_windows)):
        location = comment_windows[i].find_element_by_class_name('comment-detail-scopeline').text
        locations.append(location)
        
        
        # creating a list of authors 
    authors = []
    for i in range(len(comment_windows)):
        author = comment_windows[i].find_element_by_class_name('author-menu-box-container').text
        authors.append(author)
        comments_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])
    
    
        # creating list of dates
    dates = []
    comment_dates = driver.find_elements_by_class_name('css-9p9z55')
    for comment in comment_dates:
        dates.append(comment.text)
        
    post_ids= []
    for i in range(len(comment_windows)):
        post_id = href.split('=')[1]
        post_ids.append(post_id)
  
    comments_df = pd.DataFrame()
        # set columns of dataframe
    comments_df['post_id'] = [post_id for post_id in post_ids]
    comments_df['location'] = [loc for loc in locations]
    comments_df['date'] = [date for date in dates]
    comments_df['author'] = [auth for auth in authors] 
    comments_df['comment'] = [com for com in comments]

    return comments_df


# In[31]:


def get_content(df):   
    # PATH = './chromedriver.exe'
    # driver = webdriver.Chrome(PATH)
    
    posts_df = pd.DataFrame(columns = ['post_id','author','location','date','post'])
    comments_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])
    comments_master_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])
    
    for href in df['href']:
        #comments_temp_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])

        try:
            post = get_post(href)
            posts_df = posts_df.append(post, ignore_index=True)
            comments_df = get_comments(href)
            comments_master_df = comments_master_df.append(comments_df)
        except: 
            continue
    return posts_df, comments_master_df   


# In[60]:

def start_driver():
    url = 'https://nextdoor.com/'
    # set driver
    PATH = '../chromedriver'
    driver = webdriver.Chrome(PATH)
    return driver



def run_scrape():
    print('Ready To Scrape Nextdoor!')
    usr_email = input('What is your email:')
    pwd = input('Enter Pwd:')
    keyword = input('Input Search Term:')

#     url = 'https://nextdoor.com/'
#      # set driver
#     PATH = '../chromedriver'
#     driver = webdriver.Chrome(PATH)
    
    href_list = scrape(f'{usr_email}', f'{pwd}', f'{keyword}' )
    posts, comments = get_content(href_list)
    
    return posts, comments


# # V2
# 
# It appeared during my scrape that the nextdoor.com server noticed my activity and began responding with altered HTML and URL formats. This may have also been due to routine maintenance on the server. In either case, here is the code that was written to get around the changes when they were encountered. This will return a dataframe where the post_id is in a different format than the previous version.

# In[10]:


def scrape_2(usrnm, pwd):
    '''
    get links for given key word'''
    nxtdr_login(usrnm,pwd)
    time.sleep(3)
    links = search('homeless',100)
    time.sleep(2)
    
    links_df = pd.DataFrame()
    # fill out href column with contents of links list 
    links_df['href'] = [ link for link in links]
     # clean href 
    links_df['href'] = [ href[0] for href in links_df['href'].str.split('?')]
    # drop duplicates posts
    links_df.drop_duplicates( inplace = True,   keep='first')
    # only take user posts not ads etc.
    usr_post = links_df['href'].str.contains(f'/p/')
    links_df = links_df[usr_post]
    return links_df


# In[12]:


def get_post_2(href):
    '''
    scrape post from given link 
    '''
    # open post in browser
    driver.get(href)
           # give time to load
    time.sleep(1)
        # container for post and comments 
   # post_container = driver.find_element_by_class_name('css-1dkvlfs')
        # container for post
    main_post_container = driver.find_element_by_class_name('cee-media-body')
        # actual text of main post 
    main_post = main_post_container.find_element_by_class_name('Linkify').text
        # post id 
    post_id = href.split('/p/')[1]

    # meta info
        # location
    main_post_location = main_post_container.find_element_by_tag_name('button').text
        # author name
    meta = main_post_container.find_elements_by_tag_name('a')
    main_post_author = meta[0].text

        # post date 
        # list of entities that mess up date pull 
    date_fix_list = ['City of Denver','News','Denver Police Department']

    if main_post_author in date_fix_list:
        main_post_date = meta[2].text
    else:
        main_post_date = meta[1].text

        # post to append to post_df
    post = {'post_id': post_id ,
           'author' : main_post_author, 'date': main_post_date,
           'location': main_post_location, 'post': main_post}
    
    return post


# In[14]:


def get_comments_2(href):
    ''' scrape comments from given link'''
    time.sleep(1)   
    # clicking the seem more comments button 
    see_more_comments = driver.find_element_by_class_name('see-previous-comments-button-paged')
    see_more_comments.click()
   
        # the boxes around each of the comments on the post
    comment_window = driver.find_element_by_class_name('css-1cefqj0')
    # tag containing authors and locations 
    comments_meta = comment_window.find_elements_by_class_name('css-15h9wih')
    # actual text of comments
    comments_text = comment_window.find_elements_by_class_name('_1aEnMjGe')
    # the date of the comments
    comments_dates = comment_window.find_elements_by_class_name('css-9p9z55')
    
    
        # creating a list for all of our data types
    post_ids, dates, authors, locations, comments = [],[],[],[],[]

    # iterate through web elements and add their contents to lists 
    for comment in comments_text:
        comments.append(comment.text)

    for date in comments_dates:
        dates.append(date.text)

    for meta in comments_meta:
        meta = meta.text.split(' • ')
        author = meta[0]
        location = meta[1]
        authors.append(author)
        locations.append(location)


    for i in range(len(comments_dates)):
        post_id = href.split('/p/')[1]
        post_id = post_id.split('?')[0]
        post_ids.append(post_id)
      
    # create dataframe
    comments_df = pd.DataFrame()

    # Fill values of dataframe with list values.
    comments_df['post_id'] = [post_id for post_id in post_ids]
    comments_df['location'] = [loc for loc in locations]
    comments_df['date'] = [date for date in dates]
    comments_df['author'] = [auth for auth in authors] 
    comments_df['comment'] = [com for com in comments]

    return comments_df


# In[16]:


def get_content_2(df):   
    # PATH = './chromedriver.exe'
    # driver = webdriver.Chrome(PATH)
    
    posts_df = pd.DataFrame(columns = ['post_id','author','location','date','post'])
    comments_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])
    comments_master_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])
    
    for href in df['href']:
        
        #comments_temp_df = pd.DataFrame(columns = ['post_id','author','location','date','comment'])

        try:
            post = get_post_2(href)
            time.sleep(2)
            posts_df = posts_df.append(post, ignore_index=True)
            comments_df = get_comments_2(href)
            comments_master_df = comments_master_df.append(comments_df)
        except: 
            continue
    return posts_df, comments_master_df 


