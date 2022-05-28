import pandas as pd
import numpy as np
import nltk
import pandas as pd
import re
import os
from nltk.tokenize import word_tokenize
# 


def module_1_entities(data,uni_df,country_df, city_df):



#......................................urls regex..........................................

    def _urls(g):
        urls=[]
    #     print(data.head())
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    #     for g in data['Resumes']:
        url_matches = url_pattern.finditer(str(g))
    #     print(url_matches)
        for match in url_matches:
            urls.append(match.group(0))
        return urls

#....................................email regex.............................................
    def _email(g):
        email=[]
    #     y=""
        
    #     print(data.head())
    #     email_pattern = re.compile(r'[a-zA-Z0-9-\.]+@[a-zA-Z-\.]*\.(com|edu|net)')
        email_pattern = re.compile(r'[a-zA-Z0-9-\.]+@[a-zA-Z-\.]*\.(com|edu|net)')

    #     for g in data['Resumes']:
        email_matches = email_pattern.finditer(g)
    #     print(email_matches)
        for match in email_matches:
            email.append(match.group(0))
    #         print(match.group(1))
        return email
        
#..........................................contacts regex................................................

    def _contact(g):
        y=""
        cont=[]
        contact_pattern = re.compile(r'(?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    #     for g in data['Resumes']:
        contact_matches = contact_pattern.finditer(str(g))
        for match in contact_matches:
            cont.append(match.group(0))
        return cont

#...........................................uni details...........................................................
    def _university(g, uni_df):
        univ=[]
    #     for g in data['cleaned']:
        # uni=pd.DataFrame(pd.read_csv(r"C:\Users\WaseemAhmad\extraViz\spacy\azureFlaskApi\world_universities.csv"))

        g=str(g).lower()
        for x in uni_df['UniversityName']:
            x=str(x).lower()
            if x in g:
                univ.append(x)
        return univ

#.............................................city details....................................................................

    def _city(g,city_df):
        citi=[]

        g=str(g).lower()
        g=word_tokenize(g)
        for x in city_df['City']:
    #         print(x)
            x=str(x).lower()
            if x in g:
                citi.append(x)
        return citi
#.............................................country details.....................................................
    def _country(g,country_df):
        countri=[]

        g=str(g).lower()
        g=word_tokenize(g)
        for z in country_df['GEOGRAPHY']:
    #         print(x)
            z=str(z).lower()
            if z in g:
                countri.append(z)
        return countri

        
        # col=['Skill','Date','Location','Organization','NORP','Languages','Name','City','University','Country','Email','Contact']
    col=['City','Location','University','Contact','Email','URLS']

    new_data2=pd.DataFrame(columns=col)


    

    # uni.head()



    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', str(resumeText))  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText
    
    data['cleaned'] = data['Resumes'].apply(lambda x:cleanResume(x))

    for g in range(len(data)):
    #     print(g)
        city=_city(data['cleaned'][g],city_df)

        location=_country(data['cleaned'][g],country_df)
        # print(location)
        uni=_university(data['Resumes'][g], uni_df)
    #     print(uni)
        cont=_contact(data['Resumes'][g])   
    #     print(cont)
        email=_email(data['Resumes'][g])
    #     print(email)
        urls=_urls(data['Resumes'][g])

        new_data2=new_data2.append({'City':city,'Location':location,'University':uni,'Contact':cont,'Email':email,'URLS':urls}, ignore_index=True)


    return new_data2
