import pandas as pd
from langdetect import detect
from urllib.request import Request, urlopen
import ssl
import requests
import string
from inscriptis import get_text



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
    empty = []


    def scrapingdata(urlfiles):
        list_url = []
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

        # print(list_url)
        # start = 0

        for all_list in list_url:
            print(type(all_list), 'all list type')
            try:
                # html = urllib.request.urlopen(str(all_list)).read().decode('utf-8')

                ## Adding user agent to hide bot access
                ssl.match_hostfname = lambda cert, hostname: True
                new_url = all_list
                req = Request(new_url, headers={'User-Agent': 'Mozilla/5.0'})
                html = urlopen(req).read().decode('utf-8')
                text = get_text(html)
                language = detect(text)
                print(language)

                ## If condition matches to English then data preprocessing starts
                if language == 'en':
                    # English_url.append(all_list)
                    data = text.split()
                    no_punc = [char for char in data if char not in string.punctuation]

                    ## Stripping data
                    no_punc1 = [stripp.strip('?#{}()@%£€"[]!.\'\",:;_▼   ™®©+$-*/&|<>=~0123456789') for stripp in
                                no_punc]
                    more_words = [word1 for word1 in no_punc1 if len(no_punc1) > 2]
                    res = []
                    [res.append(x) for x in more_words if x not in res]
                    # no_punc = ''.join(no_punc)

                    ## Importing stopwords and removing from extracted data
                    ## Change path of text file
                    newStopWords = pd.read_csv("stop.txt")
                    extended_stopwords = ''.join(newStopWords['a'].tolist())
                    clean_words = [word for word in res if word.lower() not in extended_stopwords]
                    # stripped_data = [stripp for stripp in clean_words
                    # if  clean_words.strip('{}()"[]!.\'\",:;+$-*/&|<>=~0123456789')]
                    lowered = [lowercase.lower() for lowercase in clean_words]
                    # new_data = [data.strip('{}()"[]!.\'\",:;+$-*/&|<>=~0123456789') for data in lowered]
                    ## Removing digits from text
                    no_integers = [x for x in lowered if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]



                    empty.append(no_integers)
                    print("completing each loop")


                else:
                    print('Non English Url')
                    # Non_English_url.append(all_list)




            except:
                pass

    try:
        print(scrapingdata(int_features))

        df = pd.DataFrame(empty)
        # df['Data'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df['Data'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df1 = df['Data'].tolist()
        print(df1, 'df is here')

        # model,tfidf = pickle.load(open('MultiLiModel.pkl','rb'))
        # sample1 = tfidf.transform(df1)
        # pred = model.predict(sample1)

        model, tfidf, id_to_category, cat_id = pickle.load(open('MultiLiModel.pkl', 'rb'))
        sample1 = tfidf.transform(df1)
        pred = model.predict(sample1)
        print(pred)


        if pred == 0:
            out = 'Alcohol'
            output.append(out)
        elif pred == 1:
            out ='War-violence-weapon'
            output.append(out)
        elif pred == 2:
            out ='Religion'
            output.append(out)
        elif pred == 3:
            out ='Porn'
            output.append(out)
        elif pred == 4:
            out ='Drug'
            output.append(out)
        elif pred == 5:
            out ='Finance'
            output.append(out)
        elif pred == 6:
            out ='Education'
            output.append(out)
        elif pred == 7:
            out ='Search Engine'
            output.append(out)
        elif pred == 8:
            out ='Job'
            output.append(out)
        elif pred == 9:
            out ='Politics'
            output.append(out)
        elif pred == 10:
            out ='Hacking'
            output.append(out)
        elif pred == 11:
            out ='Chat/Social Media'
            output.append(out)
        elif pred == 12:
            out ='Online Shopping'
            output.append(out)
        else:
            out ="Something is wrong please check"
            output.append(out)

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
