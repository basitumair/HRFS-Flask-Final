# pip3 install PyPDF2
# pip install textract
# pip install nltk
# pip install docx2txt
#pip install fuzzywuzzy
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
#spacy
import spacy
# from spacy.pipeline import EntityRuler
# from spacy.lang.en import English
# from spacy.tokens import Doc

#gensim
# import gensim
# from gensim import corpora

#Visualization
from spacy import displacy
# import pyLDAvis.gensim_models
# from wordcloud import WordCloud
# import plotly.express as px
import matplotlib.pyplot as plt

#Data loading/ Data manipulation
import pandas as pd
import numpy as np
import jsonlines
import os
#nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])


#warning
import warnings
warnings.filterwarnings('ignore')
import PyPDF2
# import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import docx2txt as dtxt
import pandas as pd
import re
# !pip install pdfminer
import os
# position='ENGINEERING''
# from docx import *
import spacy
from flask import jsonify, make_response
#Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")



#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

from flask import Flask, redirect, url_for, request, render_template,make_response, send_file,jsonify
from json2html import *
# from IPython.display import HTML
sys.setrecursionlimit(10**6)
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import re
from io import BytesIO
import jinja2
import string
from urllib.request import urlopen
punct =  string.punctuation
punct=list(punct)
punct.remove('|')
import glob
import PyPDF2
# import fitz

from flask_cors import CORS, cross_origin
from pdfminer.high_level import extract_text
from importlib import reload
from EntitiesExtraction_module1 import *
import EntitiesExtraction_module1 as mod1
from EntitiesExtraction_module2 import *
import EntitiesExtraction_module2 as mod2
reload(mod1)
reload(mod2)
#read model libraries
import gensim
from gensim import corpora
#Visualization
from spacy import displacy
import pyLDAvis.gensim_models
# from wordcloud import WordCloud
import plotly.express as px

#read dataset
# city.head()
uni=pd.DataFrame(pd.read_csv("world_universities.csv"))
# print(uni.head())
country=pd.DataFrame(pd.read_excel("cities500.xlsx"))
city=pd.DataFrame(pd.read_excel("cities500.xlsx"))
#import ranking datasets
df = pd.read_csv("Resume.csv")
df = df.reindex(np.random.permutation(df.index))
data = df.copy().iloc[
    0:200,
]
data.head()

import spacy
nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/index2", methods=["GET"])
def index():
    """GET in server"""

    # request_data = request.get_json()
    # dataframe = getlistdataframe(request_data["DataFileUrls"])
    response = jsonify({ 'StatusCode' : 200 , 'Message': 'Newly created app' })

    return response


@app.route("/get_example", methods=["POST"])
def get_example():
    """GET in server"""
    response = jsonify({ 'StatusCode' : 200 , 'Message': 'Newly created app' })
    return response


@app.route('/Resume_segmentation', methods=['POST'])
def read_resume():
    request_data = request.get_json()
    urls = request_data['Urls']
    skills = request_data['Skills']
    country_desired = request_data['DesiredCountries']



    required_skills_list=skills.lower().split(",")
    print(f"{required_skills_list}")
    nationalaity_preference=country_desired
    print(f"@@@@@@@@@@@@{nationalaity_preference}")



    # -------------------------------------------------------------------------------------------------------------
    def read_files(urls):
        # -------------------------------------------------------------------------------------------------------------
        def extract_text_from_pdf(pdf_path):

            return extract_text(pdf_path)

        # urls = urls.replace('{', '')
        # urls = urls.replace('}', '')
        try:
            files_path=[]
            urls = urls.split(",")
            print(urls)
            for url in urls:
                f = urlopen(url=url)
                
                print("---Actual Url---", url)
                print("---url----", f)
                if f != " ":
                    html = f.read()
                    import os
                    LOCAL_BLOB_PATH = 'resume'
                    file_name = url.split("/")[4]
                    print("--File Name--", file_name)
                    download_file_path = os.path.join(LOCAL_BLOB_PATH, file_name)
                    # for nested blobs, create local path as well!
                    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

                    with open(download_file_path, "wb") as file:
                        file.write(html)
            #             read_files=(file_name)
            import glob
            resumes = []
            #         print(os.chdir("*"))
            for file in glob.glob('resume' + '/*'):
                print("file for resume list---", file)
                file_name=str(file)
                file_name=file_name.replace("\\", "")
                files_path.append(file_name)
                if file.endswith('.pdf'):
                    resumes.append(extract_text_from_pdf(file))

                elif file.endswith('.docx'):
                    text = dtxt.process(file)
                    #         file.endswith('.docx'):
                    resumes.append(text)

            return resumes,files_path
        except Exception as ex:
           print("exception---", ex)
           return json.dumps(ex)

    def get_skills(text):
        doc = nlp(text)
        myset = []
        subset = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                subset.append(ent.text)
        myset.append(subset)
        return subset

    def unique_skills(x):
        return list(set(x))

    try:
        dir = 'resume/'
        filelist = glob.glob(os.path.join(dir, "*"))
        for f in filelist:
            os.remove(f)
        # -----------------------------------------------------------------------------------------------------------------

        skills_list = skills
        skills_list = skills_list.split(",")
        # ................................................................................
        resumes,resumes_url = read_files(urls)
        data = pd.DataFrame(list(resumes), columns=['Resumes'])
        result = mod1.module_1_entities(data, uni, country, city)

        for a in range(len(result)):
            result['City'][a] = list(dict.fromkeys(result['City'][a]))
            result['Location'][a] = list(dict.fromkeys(result['Location'][a]))
            result['University'][a] = list(dict.fromkeys(result['University'][a]))
            result['Contact'][a] = list(dict.fromkeys(result['Contact'][a]))
            result['Email'][a] = list(dict.fromkeys(result['Email'][a]))
            result['URLS'][a] = list(dict.fromkeys(result['URLS'][a]))

            input_resume = resumes[a]

            resume_skills = unique_skills(get_skills(input_resume.lower()))
            score = 0
            for x in required_skills_list:
                if x in resume_skills:
                    score += 1
            req_skills_len = len(required_skills_list)
            match = round(score / req_skills_len * 100, 1)
            print(f"The current Resume is {match}% matched to your requirements")
            result['Skill'][a] = resume_skills
            result['Similarity'][a] = match
            result['Resume_Url'][a]= resumes_url[a]

        result = result.to_json()

    #     result3=result2.to_json()

    #     result3=HTML(df.to_html(classes='table table-stripped'))

    #     json_data = jsonify(trails=result),{'Content-Type':'application/json'}

        return result
    except Exception as ex:
        print("exception main ---", ex)
        return json.dumps(ex)

#     return json2html.convert(json=json_data)
#     os.chdir(r"C:\Users\WaseemAhmad\extraViz\spacy\flaskapi")
#     workingdir = os.path.abspath(os.getcwd())
#     print(workingdir)
#     return render_template("resumes_module1.html", dic=result, length1=length1,length2=length2,dic2=result2)
# ..........................................................................................................
# @app.route('/Resume_segmentation_module2',methods = ['POST', 'GET'])
# def resume_segmentation2():


def remove_files():
    import os, shutil
    folder = 'resume'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


import json

class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

if __name__ == '__main__':
    app.run()
    # app.run(host="localhost" , port=8000)
    # app.run(debug=True, host='0.0.0.0')


