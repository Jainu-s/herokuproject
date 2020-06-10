import pandas as pd
import os
import spacy
import time
import json
from langdetect import detect
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from urllib.request import Request, urlopen
import ssl
import requests
import string
from inscriptis import get_text
import pickle
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from googletrans import Translator


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__ , template_folder='.')
model = pickle.load(open('MultiLiModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',whiskey = 42)


output = []
@app.route('/predict',methods=['POST'])
def predict():
    global output

    int_features = [x for x in request.form.values()]
    print(int_features,'here is int_features')

    url_list = []
    user_input = input('Please Enter Url (eg :"website.com"): ')
    url_list.append(user_input)
    


    '''
    Creating scraping function to run two for loops. One for finding the http or https protocols and another for loop
    to data pre-process.
    '''

    empty=[]
    final_text=[]
    def scrapingdata(urlfiles):

        list_url = []

        ## This for loop is used to find http or https protocol
        for url in urlfiles:

            try:
                final_url = 'http://' + url
                r = requests.get(final_url, allow_redirects=True)
                fullurl = r.url
                protocol = fullurl[:fullurl.find(":")]
                print(protocol)

                if protocol == 'http':
                    found_url_type = 'http://' + url
                    list_url.append(found_url_type)
                elif protocol == 'https':
                    found_url_type = 'https://' + url
                    list_url.append(found_url_type)

                else:
                    found_url_type = 'http://' + url
                    list_url.append(found_url_type)
            except:
                pass



        ## This for loop is used for data scraping using Request,Curl and Selenium
        for all_list in list_url:
            print(type(all_list),'all list type')
            try:
                ## url is Assigning to new_url
                new_url = all_list


                ## Executing Request Scraping using main() method and assigning out to text1_data

                ssl.match_hostfname = lambda cert, hostname: True

                ## Useragents Added 'using random user agent' library
                software_names = [SoftwareName.CHROME.value]
                operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]
                user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems,limit=100)

                # Get Random User Agent String.
                user_agent = user_agent_rotator.get_random_user_agent()
                req = Request(new_url, headers={'User-Agent': user_agent})
                html = urlopen(req).read().decode('utf-8')
                text1 = get_text(html)
                final_text.append(text1)


                ## Finding length of scraped data from Request, If scraped data is less than 1000 words then curl start scraping
                if len(text1) < 500:
                    os.system(f"curl -L --max-time 20 -0 {new_url} -o output.txt")

                    saved_ouput = os.path.dirname(os.path.abspath(__file__))
                    output = os.path.join(saved_ouput, 'output.txt')


                    with open(output, "rb") as f:
                        soup = BeautifulSoup(f, "lxml")
                        [s.decompose() for s in soup("script")]  # remove <script> elements
                        text4 = soup.body.get_text()
                        print(text4, 'This is text 4')
                        final_text.append(text4)


                    ''' Here the nested for loop is used as continuation of curl. If curl scraped data is less than 1000 then selenium does the scraping and
                    need to check that chromedriver should be same version as chrome browser. Also chromedriver file should be in the directory
                    chromedriver link: https://chromedriver.storage.googleapis.com/index.html?path=83.0.4103.39/
                    '''
                    print(len(text4),'length of text4')
                    if len(text4) < 500:
                        cdriver = os.path.dirname(os.path.abspath(__file__))
                        chromedriver = os.path.join(cdriver, 'chromedriver')
                        driver = webdriver.Chrome(f"{chromedriver}")
                        driver.get(new_url)
                        driver.implicitly_wait(90)
                        paragraph = driver.find_elements_by_tag_name('p')
                        texts = []

                        for para in paragraph:
                            text5 = para.text
                            print(text5, 'Selenium Scraped **')
                            final_text.append(text5)
                            texts.append(text5)




                ## In scraping process data is stored in final_text list and now we make to string and assigning to removed
                removed = str(final_text)[1:-1]
                removed = removed.replace('\\n','')
                language = detect(removed)
                print(language)


                lemmat = []
                ## From here data pre-processing starts if language is detected as English
                ## If condition matches to English then data preprocessing starts
                if language == 'en':
                    ##English_url.append(all_list)
                    ## By default en_core_web_sm won't be available so download from this command: python3 -m spacy download en_core_web_sm
                    data = removed.split()
                    nlp = spacy.load("en_core_web_sm")
                    for words in data:
                        doc = nlp(words)
                        for token in doc:
                            if token.text != token.lemma_:
                                lemmat.append(token.lemma_)
                            else:
                                lemmat.append(token.text)

                    no_punc = [char for char in lemmat if char not in string.punctuation]
                    print(no_punc,'here is no punc ------------------->')

                    ## Stripping data
                    no_punc1 = [stripp.strip('?#{}()@%£€"[]!.\'\",:;_▼   ™®©+$-*/&|<>=~0123456789') for stripp in no_punc]
                    more_words = [word1 for word1 in no_punc1 if len(no_punc1) > 2]
                    res = []
                    [res.append(x) for x in more_words if x not in res]
                    print(res,'res is here --------------------------->')


                    ## Importing stopwords and removing from extracted data

                    stop_words = os.path.dirname(os.path.abspath(__file__))
                    stopwords = os.path.join(stop_words, 'stop.txt')
                    print(stopwords,'here is the path of stopwords')
                    newStopWords = pd.read_csv(stopwords)
                    extended_stopwords = ''.join(newStopWords['a'].tolist())
                    clean_words = [word for word in res if word.lower() not in extended_stopwords]

                    ## lowering the case of data scraped
                    lowered = [lowercase.lower() for lowercase in clean_words]

                    ## Removing digits from text
                    no_integers = [x for x in lowered if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

                    ## Appending entire pre-processed data to empty list
                    empty.append(no_integers)
                    print("completing each loop")


                else:

                    print('Non English Url')
                    translator = Translator()
                    text = translator.translate(removed)
                    trans_text = text.text
                    print(text.text)

                    ##English_url.append(all_list)
                    ## By default en_core_web_sm won't be available so download from this command: python3 -m spacy download en_core_web_sm
                    data = trans_text.split()
                    nlp = spacy.load("en_core_web_sm")
                    for words in data:
                        doc = nlp(words)
                        for token in doc:
                            if token.text != token.lemma_:
                                lemmat.append(token.lemma_)
                            else:
                                lemmat.append(token.text)

                    no_punc = [char for char in lemmat if char not in string.punctuation]
                    print(no_punc, 'here is no punc ------------------->')

                    ## Stripping data
                    no_punc1 = [stripp.strip('?#{}()@%£€"[]!.\'\",:;_▼   ™®©+$-*/&|<>=~0123456789') for stripp in
                                no_punc]
                    more_words = [word1 for word1 in no_punc1 if len(no_punc1) > 2]
                    res = []
                    [res.append(x) for x in more_words if x not in res]
                    print(res, 'res is here --------------------------->')

                    ## Importing stopwords and removing from extracted data

                    stop_words = os.path.dirname(os.path.abspath(__file__))
                    stopwords = os.path.join(stop_words, 'stop.txt')
                    print(stopwords, 'here is the path of stopwords')
                    newStopWords = pd.read_csv(stopwords)
                    extended_stopwords = ''.join(newStopWords['a'].tolist())
                    clean_words = [word for word in res if word.lower() not in extended_stopwords]

                    ## lowering the case of data scraped
                    lowered = [lowercase.lower() for lowercase in clean_words]

                    ## Removing digits from text
                    no_integers = [x for x in lowered if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

                    ## Appending entire pre-processed data to empty list
                    empty.append(no_integers)
                    print("completing each loop")







            except:
                pass

    '''
    The Scraping data function is completed and we are calling scrapingdata function and now the data stored in empty list
    is created as dataframe which can also be exported if necessary. Now the list words in df dataframe is joined and converted
    to list
    '''

    try:
        print(scrapingdata(url_list))
        df = pd.DataFrame(empty)
        df['Data'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df1 = df['Data'].tolist()

        ## Here we importing already trained model pickle file to predict the data that is scraped from domain.
        model = os.path.dirname(os.path.abspath(__file__))
        pred_model = os.path.join(model, 'MultiLiModel.pkl')
        model, tfidf = pickle.load(open(pred_model, 'rb'))
        sample1 = tfidf.transform(df1)
        pred = model.predict(sample1)


        if pred == 0:
            print('Alcohol')
        elif pred == 1:
            print('War-violence-weapon')
        elif pred == 2:
            print('Religion')
        elif pred == 3:
            print('Porn')
        elif pred == 4:
            print('Drug')
        elif pred == 5:
            print('Finance')
        elif pred == 6:
            print('Education')
        elif pred == 7:
            print('Search Engine')
        elif pred == 8:
            print('Job')
        elif pred == 9:
            print('Politics')
        elif pred == 10:
            print('Hacking')
        elif pred == 11:
            print('Chat/Social Media')
        elif pred == 12:
            print('Online Shopping')
        else:
            print("Something is wrong please check")

    except ValueError:
        print("Something is wrong with your url please try with another url.")
        print("Also check:")
        print("1.As the website has more text or not")
    else:
        print("Everything went successfully.")




        #prediction = model.predict(int_features)

        #out = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Category Predicted {}'.format(output))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    out = prediction[0]
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)