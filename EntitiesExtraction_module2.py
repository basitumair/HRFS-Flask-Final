import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import sys
sys.setrecursionlimit(3000)
sys.setrecursionlimit(10**6)
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

#Import the requisite library
import spacy

#Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")



#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")





import docx2txt
import re
# def Email()
pattern = re.compile(r'[a-zA-Z0-9-\.]+@[a-zA-Z-\.]*\.(com|edu|net)')

def best_Match_resume(data, skills_list):

    # for g in data['Resumes']:
    # #     print(g)
    # #     doc = nlp(g)
    #     matches = pattern.finditer(str(g))
    #     #     print(matches)
    #     for match in matches:
    # #         print(match)
    #         print(match.group(0))


    # print("skills list: ",skills_list)



    def _skill(data):
        # 'Location','Organization',

                    # col=['Skill','Date','Location','Organization','NORP','Languages','Name','City','University','Country','Email','Contact']
        col=['Skill','Date','summary']

        new_data1=pd.DataFrame(columns=col)
            # lis=[]
        # for a in patterns:
        #     lis.append(a)
        # # print(lis)
        # ruler.add_patterns(lis)

        # def entities():
        ruler.from_disk(r"jz_skill_patterns.jsonl")
        skills=[]
        phone=[]
        # Loc=[]
        # org=[]
        # norp=[]
        # language=[]
        summary=[]
        for g in data['cleaned']:
            doc = nlp(g)

        #     print(doc[:10])
            summary=doc[:20]

        #     skills
            #extract entities
        #     break
            skills=[]
            phone=[]
            # Loc=[]
            # org=[]
            # norp=[]
            # language=[]
            for ent in doc.ents:
        #         print (ent.text, ent.label_)
                if ent.label_=='SKILL':
        #             print(ent.text)
                    skills.append(ent.text)
                if ent.label_=='DATE':
        #             print(ent.text)
                    phone.append(ent.text)
        #         if ent.label_=='GPE':
        # #             print(ent.text)
        #             Loc.append(ent.text)
        #         if ent.label_=='ORG':
        # #             print(ent.text)
        #             org.append(ent.text)
        # 'Location':Loc,'Organization':org,
                # if ent.label_=='NORP':
        #             print(ent.text)
                    # norp.append(ent.text)
                # if ent.label_=='LANGUAGE':
        #             print(ent.text)
                    # language.append(ent.text)
            new_data1=new_data1.append({'Skill':skills,'Date':phone,'summary':summary}, ignore_index=True)

        #     displacy.render(doc,style="ent",jupyter=True)  
            print("----Next Resume----")
    #             break
    #             data=data.append('')
        return new_data1



    #............................................skill Methos call..............................................................
    new_data1=_skill(data)
    #.............................................................................................................................
    
    #-----------------------------------------------Skills Matching Start-------------------------------------------------------
    # new_data1['NewSkills']=""
    # new_data1['match_Item']=""
    # new_data1['match_Item_Count']=""

    # for a in range(len(new_data1)):
    #     # print(new_data1['Skill'][a])
    #     # print(list(dict.fromkeys(new_data1['Skill'][a])))
    #     new_data1['NewSkills'][a]=list(dict.fromkeys(new_data1['Skill'][a]))


    # def resume_selection(new_data1,skills_list):
    #     for b in range(len(new_data1)):
    #         match_item=[]
    #         for g in new_data1['NewSkills'][b]:
    #             j=g.strip().lower()

    #             for a in skills_list:
    #                 c=a.strip().lower()
    #                 res=fuzz.ratio(c,j)
    #                 if res>=70:
    #     #                 count=count+1
    #                     match_item.append(a)
    #     #                 match_item_count.append(count)
    #         new_data1['match_Item'][b]=match_item
    #         match_item_count=len(match_item)
    #         new_data1['match_Item_Count'][b]=match_item_count
    #     return  new_data1

    # result2=resume_selection(new_data1,skills_list)

    return new_data1


