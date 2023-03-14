from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class ScrapTool:

    def visit_url(self, website_url):
        '''
        Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
        '''
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        content = requests.get(website_url,timeout=60).content
        
        #lxml is apparently faster than other settings.
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "website_text": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+
                                                               self.get_text_content(soup)
        }
        
        #Convert to Series object and return
        return pd.Series(result)
    
    def get_website_name(self,website_url):
        '''
        Example: returns "google" from "www.google.com"
        '''
        return "".join(urlparse(website_url).netloc.split(".")[-2])
    
    def get_html_title_tag(self,soup):
        '''Return the text content of <title> tag from a webpage'''
        return '. '.join(soup.title.contents)
    
    def get_html_meta_tags(self,soup):
        '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
        tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
        return ' '.join(content)
    
    def get_html_heading_tags(self,soup):
        '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
        tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)
    
    def get_text_content(self,soup):
        '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
        tags = soup.find_all(string=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                and isinstance(tag, bs4.element.Comment)==False\
                and not stripped_tag.isnumeric()\
                and len(stripped_tag)>0:
                result.append(stripped_tag)
        return ' '.join(result)

import spacy as sp
from collections import Counter
sp.prefer_gpu()
# import en_core_web_sm
#anconda prompt ko run as adminstrator and copy paste this:python -m spacy download en

# nlp = sp.load("en_core_web_sm")
import en_core_web_sm

nlp = en_core_web_sm.load()



import re
def clean_text(doc):
    '''
    Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them
    '''
    doc = nlp(doc)
    tokens = []
    exclusion_list = ["nan"]
    for token in doc:
        if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum()==False) or token.text in exclusion_list :
            continue
        token = str(token.lemma_.lower().strip())
        tokens.append(token)
    return " ".join(tokens) 

# tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), stop_words='english',
#                 # sublinear_tf=True)
# print(tfidf)

website='https://leetcode.com/problemset/all/'
scrapTool = ScrapTool()

id_to_category={0: 'Travel',1: 'Social Networking and Messaging',2: 'News',3: 'Streaming Services',4: 'Sports',5: 'Photography',6: 'Law and Government',7: 'Health and Fitness',8: 'Games',9: 'E-Commerce',10: 'Forums',11: 'Food',12: 'Education',13: 'Computers and Technology',14: 'Business/Corporate',15: 'Adult'}
cat=['Travel', 'Social Networking and Messaging', 'News',
       'Streaming Services', 'Sports', 'Photography',
       'Law and Government', 'Health and Fitness', 'Games', 'E-Commerce',
       'Forums', 'Food', 'Education', 'Computers and Technology',
       'Business/Corporate', 'Adult']

cat= np.array(cat).astype('object')
m1 = pickle.load(open('model.sav', 'rb'))


# try:
web=dict(scrapTool.visit_url(website))

text=(clean_text(web['website_text']))
# text=[text]
# print(text)

df=pd.read_csv("data.csv")

# X=df['X variable']

# print(X)
X = df['cleaned_website_text'] 
fitted_vectorizer=TfidfVectorizer(min_df=5, ngram_range=(1, 2), stop_words='english',sublinear_tf=True)



X = df['cleaned_website_text'] # Collection of text
y = df['Category'] # Target or the labels we want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

fit_tfidf = fitted_vectorizer.fit([X_train]).toarray()
print(fit_tfidf)

vectorizer = TfidfVectorizer(stop_words = my_stopwords, use_idf = True)
bag_of_words = vectorizer.fit_transform(X_new)


# We transform each cleaned_text into a vector


# labels = df.category_id


# t=fitted_vectorizer.fit_transform([text])
# # t=fitted_vectorizer.transform(text)

# print(t)
# features=tfidf.fit_transform(text)
# print(features)
# fit_tfidf.fit(text)
# t=fit_tfidf.transform([text])
# print(t)
# print(id_to_category[m1.predict(t)[0]])
# data=pd.DataFrame(m1.predict_proba(t)*100,columns=cat)
# data=data.T
# data.columns=['Probability']
# data.index.name='Category'
# a=data.sort_values(['Probability'],ascending=False)
# a['Probability']=a['Probability'].apply(lambda x:round(x,2))

# except:
    # print("Connection Timedout!")