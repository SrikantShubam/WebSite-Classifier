from flask import Flask, render_template, request, send_file

from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.model_selection import train_test_split
import spacy as sp
import psutil


# sp.prefer_gpu()
from sklearn.calibration import CalibratedClassifierCV
import joblib


# sp.prefer_gpu()
from sklearn.svm import LinearSVC
import en_core_web_sm
nlp = en_core_web_sm.load()

m1= joblib.load('linear_svc_model.joblib')
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')
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
scrapTool = ScrapTool()


app = Flask(__name__)
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
@app.route("/")
def hello_world():

    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def submit():
    
   
    site=request.form['site']
   
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'data.csv')
    df=pd.read_csv(file_path,low_memory=True)
    df['category_id'] = df['Category'].factorize()[0]
  
   
    X_train, _ = train_test_split(df['cleaned_website_text'], test_size=0.20, random_state = 0)


    
    
    tfidf.fit_transform(X_train).toarray()
    

   
    

    web=dict(scrapTool.visit_url(site))
    
    text=(clean_text(web['website_text']))
   
    t=tfidf.transform([text]).toarray()
   
    data=pd.DataFrame(m1.predict_proba(t)*100,columns=df['Category'].unique()).T
    
    data.columns=['Probability']
    data.index.name='Category'
    data=data.sort_values('Probability', ascending=False)
    print(data,type(data["Probability"]))
   




   
  

    
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss} bytes")

    return render_template('predict.html',data=data) 
    # return render_template('predict.html')

if __name__ == "__main__":
    # app.run(debug=False, use_reloader=True,host='0.0.0.0')
    app.run(port=5000)
