# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:03:06 2022

@author: joklf
"""
#### Importing packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from urllib.parse import urlparse
from tld import get_tld
import tldextract
import os.path
from nltk.tokenize import RegexpTokenizer
import re
import time
import numpy as np
import math
import seaborn as sns

## Main configuration
st.set_page_config(
    page_title="URLegit",
    page_icon="😎",
    layout="centered")

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("df.csv")
    return(data.dropna())


@st.cache(allow_output_mutation=True)
def entropy(string):
    "Calculates the Shannon entropy of a string"
    
    # get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

    # calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

    return entropy

@st.cache(allow_output_mutation = True)
def load_model():
    filename= "FinalModel.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

@st.cache(allow_output_mutation = True)
def load_scaler():
    filename = "Scaler.sav"
    loaded_scaler = pickle.load(open(filename, "rb"))
    return(loaded_scaler)



FinalModel = load_model() 

Data = load_data()
scaler =load_scaler()

### Start Webapp

st.title("URLegit  -  Schütze dich vor gefährlichen URLs")
st.markdown("Ist Ihre URL gefährlich oder können Sie sie sicher aufrufen? Lassen Sie ihre URL in vier Kategorien klassifizieren!")

#Name = st.text_input("Enter your name")
#st.write("Welcome on our Website " + Name)

st.header("Prüfen Sie Ihre URL")




with st.form("URL"):
        url = st.text_input("URL mit Protokol (https:// oder http:// ) hier einfügen:")
        submitted = st.form_submit_button("Easy Report")
        submitted2 = st.form_submit_button("Advanced Report")
        
        df = pd.DataFrame(columns = ["Arguments_LongestWordLength", "argDomanRatio", "ArgUrlRatio", "argPathRatio", "ArgLen", 
                                     "avgdomaintokenlen", "longdomaintokenlen", "tld", "delimeter_Domain", "domainlength",
                                     "domain_token_count", "domain_digit_count", "domain_letter_count", "domainUrlRatio", "Domain_LongestWordLength",
                                     "Directory_DigitCount", "Directory_LongestWordLength", "Directory_LetterCount", "Entropy_Afterpath", 
                                     "Entropy_Domain", "Entropy_DirectoryName", "Entropy_URL", "Entropy_Filename", 
                                     "File_name_DigitCount", "Filename_LetterCount", "NumberofDotsinURL", "NumberRate_Domain", 
                                     "NumberRate_FileName", "NumberRate_URL", "NumberRate_DirectoryName", "avgpathtokenlen", "pathLength", 
                                     "path_token_count", "pathDomainRatio", "pathurlRatio", "Path_LongestWordLength", "LongestPathTokenLength", 
                                     "delimeter_path", "Query_LetterCount", "Querylength", "Query_DigitCount", "SymbolCount_URL",
                                     "SymbolCount_Domain", "SymbolCount_FileName", "SymbolCount_Afterpath", "SymbolCount_Directoryname", 
                                     "urlLen", "URL_Letter_Count", "URL_DigitCount", "delimeter_url", "vowel_ratio_url"])
        
        if submitted:
            
            with st.spinner('Just a second...'):
                time.sleep(2)
                
    
        ############################################
        ###############  Features   ################
        ############################################
    
            #Domain
            domain = urlparse(url).netloc
            try:
              tld = get_tld(url, fix_protocol=True)
            except: 
              tld = str()
            subdomain = tldextract.extract(url).subdomain
            
            #Path
            path = urlparse(url).path
            filename = os.path.basename(path)
            directory = os.path.dirname(path)
            query = urlparse(url).query
            arguments = query
            afterpath = urlparse(url).params + urlparse(url).query + urlparse(url).fragment
            
            #Tokenizer and Tokens
            tokenizer = RegexpTokenizer(r'\w+')
            
            #Arguments_LongestWordLength
            tokens_arguments = tokenizer.tokenize(arguments)
            word_filter_arguments = list(filter(str.isalpha, tokens_arguments))
            if not word_filter_arguments: 
              Arguments_LongestWordLength = 0
            else: 
              Arguments_LongestWordLength = len(max(word_filter_arguments, key=len))
              
            #argDomanRatio
            if len(domain) != 0:
              argDomanRatio = len(arguments)/len(domain)
            else: 
              argDomanRatio = np.nan
              
            #ArgUrlRatio
            if len(url) != 0:
              ArgUrlRatio = len(arguments)/len(url)
            else: 
              ArgUrlRatio = np.nan
              
            #argPathRatio
            if len(path) != 0:
              argPathRatio = len(arguments)/len(path)
            else: 
              argPathRatio = np.nan
              
            #ArgLen
            ArgLen = len(arguments)
            
            #avgdomaintokenlen
            tokens_domain = tokenizer.tokenize(domain)
            tokens_length = 0
            for token in tokens_domain: 
              tokens_length += len(token)
            if len(tokens_domain) != 0:
              avgdomaintokenlen = tokens_length / len(tokens_domain)
            else:
              avgdomaintokenlen = np.nan
              
            #longdomaintokenlen
            if not tokens_domain:
              longdomaintokenlen = 0
            else:
              longdomaintokenlen = len(max(tokens_domain, key=len))
              
            #tld 
            tld = len(tld)
            
            #delimeter_Domain #Frage ob da der Punkt auch mit reinzählt- in meiner Lösung ja
            delimeter_Domain = (len(re.split(r'\W+', domain))-1)
            
            #domainlength
            domainlength = len(domain)
            
            #domain_token_count
            domain_token_count = len(tokens_domain) 
            
            #domain_digit_count
            domain_digit_count = 0
            for char in domain: 
              if char.isnumeric():
                domain_digit_count += 1
                
            #domain_letter_count (Host = Domain)
            domain_letter_count = 0
            for letter in domain: 
              if letter.isalpha():
                domain_letter_count += 1
                
            #domainUrlRatio
            if len(url) != 0:
              domainUrlRatio = len(domain)/len(url)
            else:
              domainUrlRatio = np.nan
              
            #Domain_LongestWordLength
            word_filter_domain = list(filter(str.isalpha, tokens_domain))
            if not word_filter_domain: 
              Domain_LongestWordLength = 0
            else: 
              Domain_LongestWordLength = len(max(word_filter_domain, key=len))
              
            #Directory_DigitCount
            Directory_DigitCount = 0
            for char in directory: 
              if char.isnumeric():
                Directory_DigitCount += 1
                
            #Directory_LongestWordLength 
            tokens_directory = tokenizer.tokenize(directory)
            word_filter_directory = list(filter(str.isalpha, tokens_directory))
            if not word_filter_directory: 
              Directory_LongestWordLength = 0
            else: 
              Directory_LongestWordLength = len(max(word_filter_directory, key=len))
              
            #Directory_LetterCount
            Directory_LetterCount = 0
            for letter in directory:
              if letter.isalpha():
                Directory_LetterCount += 1
                
            #Entropy_Afterpath
            Entropy_Afterpath = entropy(afterpath)
            
            #Entropy_Domain
            Entropy_Domain =  entropy(domain)
            
            #Entropy_DirectoryName
            Entropy_DirectoryName = entropy(directory)
            
            #Entropy_URL
            Entropy_URL = entropy(url)
            
            #Entropy_Filename
            Entropy_Filename = entropy(filename)
            
            #File_name_DigitCount
            File_name_DigitCount = 0
            for char in filename: 
              if char.isnumeric():
                File_name_DigitCount += 1
                
            #Filename_LetterCount
            Filename_LetterCount = 0
            for letter in filename:
              if letter.isalpha():
                Filename_LetterCount += 1
                
            #NumberofDotsinURL
            NumberofDotsinURL = url.count(".")
            
            #NumberRate_Domain
            number_domain = 0 
            for char in domain:
              if char.isdigit():
                number_domain += 1
            if len(domain) != 0:
              NumberRate_Domain = number_domain / len(domain)
            else: 
              NumberRate_Domain = np.nan
              
            #NumberRate_FileName
            Number_filename = 0 
            for char in filename:
              if char.isdigit():
                Number_filename += 1
            if len(filename) != 0:
              NumberRate_FileName = Number_filename / len(filename)
            else:
              NumberRate_FileName = np.nan
              
            #NumberRate_URL
            number_url = 0
            for char in url:
              if char.isdigit():
                number_url += 1
            if len(url) != 0:
              NumberRate_URL = number_url / len(url)
            else: 
              NumberRate_URL = np.nan 
              
            #NumberRate_DirectoryName
            number_directory = 0 
            for char in directory:
              if char.isdigit():
                number_directory += 1
            if len(directory) != 0:
              NumberRate_DirectoryName = number_directory / len(directory)
            else:
              NumberRate_DirectoryName = np.nan
              
            #avgpathtokenlen
            tokens_path = tokenizer.tokenize(path)
            tokens_length = 0
            for token in tokens_path: 
              tokens_length += len(token)
            if len(tokens_path) != 0:
              avgpathtokenlen = tokens_length / len(tokens_path)
            else:
              avgpathtokenlen = np.nan
              
            #pathLength
            pathLength = len(path)
            
            #path_token_count
            path_token_count = len(tokens_path)
            
            #pathDomainRatio
            if len(domain) != 0:
              pathDomainRatio = len(path)/len(domain)
            else:
              pathDomainRatio = np.nan
              
            #pathurlRatio
            if len(url) != 0:
              pathurlRatio = len(path)/len(url)
            else: 
              pathurlRatio = np.nan
              
            #Path_LongestWordLength
            word_filter_path = list(filter(str.isalpha, tokens_path))
            if not word_filter_path: 
              Path_LongestWordLength = 0
            else: 
              Path_LongestWordLength = len(max(word_filter_path, key=len))
              
            #LongestPathTokenLength
            if tokens_path:
              LongestPathTokenLength = len(max(tokens_path, key=len))
            else:
              LongestPathTokenLength = 0
              
            #delimeter_path 
            delimeter_path = (len(re.split(r'\W+', path))-1)
            
            #Query_LetterCount
            Query_LetterCount = 0
            for char in query: 
              if char.isalpha():
                Query_LetterCount += 1
                
            #Querylength
            Querylength = len(query)
            
            #Query_DigitCount
            Query_DigitCount = 0
            for char in query: 
              if char.isnumeric():
                Query_DigitCount += 1
                
            #SymbolCount_URL
            SymbolCount_URL = 0 
            for char in url:
              if char.isalnum() == False:
                SymbolCount_URL += 1
                
            #SymbolCount_Domain
            SymbolCount_Domain = 0 
            for char in domain:
              if char.isalnum() == False:
                SymbolCount_Domain += 1
                
            #SymbolCount_FileName
            SymbolCount_FileName = 0 
            for char in filename:
              if char.isalnum() == False:
                SymbolCount_FileName += 1
                
            #SymbolCount_Afterpath
            SymbolCount_Afterpath = 0
            for char in afterpath:
              if char.isalnum() == False:
                SymbolCount_Afterpath += 1
                
            #SymbolCount_Directoryname
            SymbolCount_Directoryname = 0
            for char in directory:
              if char.isalnum() == False:
                SymbolCount_Directoryname += 1
                
            #urlLen
            urlLen = len(url)
            
            #URL_Letter_Count
            URL_Letter_Count = 0
            for letter in url: 
              if letter.isalpha():
                URL_Letter_Count += 1
                
            #URL_DigitCount
            URL_DigitCount = 0
            for char in url: 
              if char.isnumeric():
                URL_DigitCount += 1
                
            #delimeter_url
            delimeter_url = (len(re.split(r'\W+', url))-1)
            
            #vowel_ratio_url
            num_vowels = 0
            for char in url:
              if char in "aeiouAEIOU":
                num_vowels += 1 
            if len(url) != 0:
              vowel_ratio_url = num_vowels / len(url)
            else:
              vowel_ration_url = np.nan
        
            df.loc[len(df.index)] = [Arguments_LongestWordLength, argDomanRatio, ArgUrlRatio, argPathRatio, ArgLen, 
                                 avgdomaintokenlen, longdomaintokenlen, tld, delimeter_Domain, domainlength,
                                 domain_token_count, domain_digit_count, domain_letter_count, domainUrlRatio, Domain_LongestWordLength,
                                 Directory_DigitCount, Directory_LongestWordLength, Directory_LetterCount, Entropy_Afterpath, 
                                 Entropy_Domain, Entropy_DirectoryName, Entropy_URL, Entropy_Filename, 
                                 File_name_DigitCount, Filename_LetterCount, NumberofDotsinURL, NumberRate_Domain, 
                                 NumberRate_FileName, NumberRate_URL, NumberRate_DirectoryName, avgpathtokenlen, pathLength, 
                                 path_token_count, pathDomainRatio, pathurlRatio, Path_LongestWordLength, LongestPathTokenLength, 
                                 delimeter_path, Query_LetterCount, Querylength, Query_DigitCount, SymbolCount_URL,
                                 SymbolCount_Domain, SymbolCount_FileName, SymbolCount_Afterpath, SymbolCount_Directoryname, 
                                 urlLen, URL_Letter_Count, URL_DigitCount, delimeter_url, vowel_ratio_url]
            
            df = scaler.transform(df)
            
            
            Klasse = FinalModel.predict(df)
            
            Probability = FinalModel.predict_proba(df)
            
            st.success("Your URL has been classified")
            
            for x in [Klasse]:
                        if x == 0:
                               st.success("Es handelt sich mit %i Prozent Sicherheit um einen unschädlichen URL!" % (Probability[:,0]*100))
                        if x == 1:
                               st.error("Es handelt sich mit %i Prozent Sicherheit um Malware!" % (Probability[:,1]*100))
                               st.write("Was ist Malware?")
                               st.caption("Malware ist ein Oberbegriff für verschiedene Formen von Schadsoftware, welche durch das Aufrufen des Links auf einem Gerät installiert werden. Die Installation erfolgt meist unbemerkt, arbeitet unauffällig im Hintergrund weiter und kann je länger die Infektion unentdeckt bleibt weitreichende Konsequenzen haben, wobei nicht nur PCs, sondern auch Tablets oder Smartphones betroffen sein können. Die bekanntesten Ausprägungen von Malware sind Trojaner oder Viren, weitere können den Betroffenen im schlimmsten Fall den Zugriff auf ihre Geräte gegen Lösegeld verweigern, durch das Sammeln von Informationen Identitätsdiebstahl veranlassen oder sich auf weitere Geräte ausbreiten. ")
                               st.write("Wie kann ich wissen, ob meine Geräte infiltriert wurden und wie gehe ich dagegen vor?")
                               st.caption("Dass das Gerät auffällig langsam operiert, sich der Speicher rasant füllt oder unerwartet Fenster geöffnet werden, können Indikatoren für Malware sein. Da dies aber auch gewöhnliche Symptome einer alternden Maschine sind, sollten sie nicht prinzipiell ein Grund zur Beunruhigung sein. Es kann dennoch auch präventiv sinnvoll sein, eine Antivirensoftware zum regelmässigen Scannen und in den meisten Fällen anschliessendem Entfernen von Malware zu installieren. Zwar tritt beispielsweise iOS Malware selten auf, dennoch kann Anti-Malware-Schutz nie zu 100 % gewährleistet werden. Oft handelt es sich bei Cybersicherheitssoftware um kostenpflichtige Angebote, wobei Nutzen-Kosten bzw. Schadensabwägungen getroffen werden sollten.")
                               st.caption("Quelle: https://www.avast.com/de-de/c-malware#topic-5")
                               
                        if x == 2:
                                st.error("Es handelt sich mit %i Prozent Sicherheit um Defacement!" % (Probability[:,2]*100))
                                st.write("Was ist Defacement?")
                                st.caption("Als eine der bekanntesten Angriffstechniken versuchen Hacker und Hackerinnen beim Defacement, Websites zu verfälschen und Organisationen zu schaden. Die übersetzte „Verunstaltung“ zielt häufig auf die Veränderung des Inhalts und des Erscheinungsbildes einer gesamten Website, wobei die böswillige Partei häufig die Reichweite einer Homepage zum Kundtun einer politischen oder religiösen Botschaft ausnutzen möchte.")
                                st.write("Wie bin ich von Defacement betroffen?")
                                st.caption("Aus Konsumentenperspektive gibt es kaum Massnahmen zur Präventation des Aufrufens einer defaced Website. Schliesslich könnte selbst Google wie 2012 in Rumänien zum Opfer der Methode werden, was anschliessend zu einer internen Wartung der Website führt und die Kundschaft insofern nicht weiter beeinflusst. Riskanter werden fehlleitende Informationen auf offiziellen Homepages der Regierung oder zudem verborgene Malware. Es gilt wie auch bei den anderen Klassifizierungen von URLegit, wachsam zu bleiben und etwaige Fälle zu melden.")
                                st.caption("Quelle: https://www.imperva.com/learn/application-security/website-defacement-attack/")   
                                
                        if x == 3:
                                st.error("Es handelt sich mit %i Prozent Sicherheit um Phishing!" % (Probability[:,3]*100))
                                st.write("Was ist Phishing?")
                                st.caption("„Sie haben gewonnen“, „Es gab ein Problem mit Ihrer Rechnung“ oder eine dringliche Nachricht der Bank sind typische Beispiele für versuchtes Phishing. Im Zuge des Telekommunikationsbetrugs versucht ein vermeintlich vertrauenswürdiger Angreifer, den Nutzer zum Aufrufen eines Links oder zum Herunterladen einer Datei zu bewegen, woraufhin dieser zur Aufführung privater Informationen wie Kreditkartendetails überlisten werden soll. Mithilfe verschiedener Kanäle und Strategien kann die Methode je nach Kompetenzen und Motiven der Cyberkriminellen weitreichende Konsequenzen bis hin zu Identitätsdiebstahl haben, wobei auch Unternehmen zum Opfer von Spionage werden können. ")
                                st.write("Wie kann ich wissen, ob ich betroffen bin und wie gehe ich dagegen vor?")
                                st.caption("Cybersicherheitssoftware ist dazu in der Lage, über Scannen auf Phishing aufmerksam zu machen und Werbeblocker einzusetzen. Grundsätzlich kann bei Phishing aber auch die Prävention äusserst effektiv sein: Das regelmässige Ändern von Passwörtern und eine generelle Vorsicht und Skepsis im Umgang mit potenziellen Fällen kann in Schulungsprogrammen von Sicherheitsteams erlernt werden.")
                                st.caption("Quelle: https://www.avast.com/de-de/c-phishing")
                                
                        
            
            st.write("Disclaimer")                    
            st.caption("Wir möchten maximale Sicherheit und Vorsicht im Umgang mit Cybersicherheit fördern. Aus diesem Grund haben wir bei der Programmierung und Auswahl unserer Modelle stets die Präzision maximiert. Das heisst, dass wir Links eher als z.B. Malware einstufen, obwohl sie Benign sind, anstatt zu risikieren, dass ein als Benign klassifzierter Link eigentlich Malware ist. Nichtsdestotrotz übernehmen wir keine Haftung für allfällige Schäden, die aufgrund eines von URLegit falsch klassifizierten Links entstehen.")
      
            
    
        if submitted2:
            
            with st.spinner('Just a second...'):
                time.sleep(2)
                
    
        ############################################
        ###############  Features   ################
        ############################################
    
            #Domain
            domain = urlparse(url).netloc
            try:
              tld = get_tld(url, fix_protocol=True)
            except: 
              tld = str()
            subdomain = tldextract.extract(url).subdomain
            
            #Path
            path = urlparse(url).path
            filename = os.path.basename(path)
            directory = os.path.dirname(path)
            query = urlparse(url).query
            arguments = query
            afterpath = urlparse(url).params + urlparse(url).query + urlparse(url).fragment
            
            #Tokenizer and Tokens
            tokenizer = RegexpTokenizer(r'\w+')
            
            #Arguments_LongestWordLength
            tokens_arguments = tokenizer.tokenize(arguments)
            word_filter_arguments = list(filter(str.isalpha, tokens_arguments))
            if not word_filter_arguments: 
              Arguments_LongestWordLength = 0
            else: 
              Arguments_LongestWordLength = len(max(word_filter_arguments, key=len))
              
            #argDomanRatio
            if len(domain) != 0:
              argDomanRatio = len(arguments)/len(domain)
            else: 
              argDomanRatio = np.nan
              
            #ArgUrlRatio
            if len(url) != 0:
              ArgUrlRatio = len(arguments)/len(url)
            else: 
              ArgUrlRatio = np.nan
              
            #argPathRatio
            if len(path) != 0:
              argPathRatio = len(arguments)/len(path)
            else: 
              argPathRatio = np.nan
              
            #ArgLen
            ArgLen = len(arguments)
            
            #avgdomaintokenlen
            tokens_domain = tokenizer.tokenize(domain)
            tokens_length = 0
            for token in tokens_domain: 
              tokens_length += len(token)
            if len(tokens_domain) != 0:
              avgdomaintokenlen = tokens_length / len(tokens_domain)
            else:
              avgdomaintokenlen = np.nan
              
            #longdomaintokenlen
            if not tokens_domain:
              longdomaintokenlen = 0
            else:
              longdomaintokenlen = len(max(tokens_domain, key=len))
              
            #tld 
            tld = len(tld)
            
            #delimeter_Domain #Frage ob da der Punkt auch mit reinzählt- in meiner Lösung ja
            delimeter_Domain = (len(re.split(r'\W+', domain))-1)
            
            #domainlength
            domainlength = len(domain)
            
            #domain_token_count
            domain_token_count = len(tokens_domain) 
            
            #domain_digit_count
            domain_digit_count = 0
            for char in domain: 
              if char.isnumeric():
                domain_digit_count += 1
                
            #domain_letter_count (Host = Domain)
            domain_letter_count = 0
            for letter in domain: 
              if letter.isalpha():
                domain_letter_count += 1
                
            #domainUrlRatio
            if len(url) != 0:
              domainUrlRatio = len(domain)/len(url)
            else:
              domainUrlRatio = np.nan
              
            #Domain_LongestWordLength
            word_filter_domain = list(filter(str.isalpha, tokens_domain))
            if not word_filter_domain: 
              Domain_LongestWordLength = 0
            else: 
              Domain_LongestWordLength = len(max(word_filter_domain, key=len))
              
            #Directory_DigitCount
            Directory_DigitCount = 0
            for char in directory: 
              if char.isnumeric():
                Directory_DigitCount += 1
                
            #Directory_LongestWordLength 
            tokens_directory = tokenizer.tokenize(directory)
            word_filter_directory = list(filter(str.isalpha, tokens_directory))
            if not word_filter_directory: 
              Directory_LongestWordLength = 0
            else: 
              Directory_LongestWordLength = len(max(word_filter_directory, key=len))
              
            #Directory_LetterCount
            Directory_LetterCount = 0
            for letter in directory:
              if letter.isalpha():
                Directory_LetterCount += 1
                
            #Entropy_Afterpath
            Entropy_Afterpath = entropy(afterpath)
            
            #Entropy_Domain
            Entropy_Domain =  entropy(domain)
            
            #Entropy_DirectoryName
            Entropy_DirectoryName = entropy(directory)
            
            #Entropy_URL
            Entropy_URL = entropy(url)
            
            #Entropy_Filename
            Entropy_Filename = entropy(filename)
            
            #File_name_DigitCount
            File_name_DigitCount = 0
            for char in filename: 
              if char.isnumeric():
                File_name_DigitCount += 1
                
            #Filename_LetterCount
            Filename_LetterCount = 0
            for letter in filename:
              if letter.isalpha():
                Filename_LetterCount += 1
                
            #NumberofDotsinURL
            NumberofDotsinURL = url.count(".")
            
            #NumberRate_Domain
            number_domain = 0 
            for char in domain:
              if char.isdigit():
                number_domain += 1
            if len(domain) != 0:
              NumberRate_Domain = number_domain / len(domain)
            else: 
              NumberRate_Domain = np.nan
              
            #NumberRate_FileName
            Number_filename = 0 
            for char in filename:
              if char.isdigit():
                Number_filename += 1
            if len(filename) != 0:
              NumberRate_FileName = Number_filename / len(filename)
            else:
              NumberRate_FileName = np.nan
              
            #NumberRate_URL
            number_url = 0
            for char in url:
              if char.isdigit():
                number_url += 1
            if len(url) != 0:
              NumberRate_URL = number_url / len(url)
            else: 
              NumberRate_URL = np.nan 
              
            #NumberRate_DirectoryName
            number_directory = 0 
            for char in directory:
              if char.isdigit():
                number_directory += 1
            if len(directory) != 0:
              NumberRate_DirectoryName = number_directory / len(directory)
            else:
              NumberRate_DirectoryName = np.nan
              
            #avgpathtokenlen
            tokens_path = tokenizer.tokenize(path)
            tokens_length = 0
            for token in tokens_path: 
              tokens_length += len(token)
            if len(tokens_path) != 0:
              avgpathtokenlen = tokens_length / len(tokens_path)
            else:
              avgpathtokenlen = np.nan
              
            #pathLength
            pathLength = len(path)
            
            #path_token_count
            path_token_count = len(tokens_path)
            
            #pathDomainRatio
            if len(domain) != 0:
              pathDomainRatio = len(path)/len(domain)
            else:
              pathDomainRatio = np.nan
              
            #pathurlRatio
            if len(url) != 0:
              pathurlRatio = len(path)/len(url)
            else: 
              pathurlRatio = np.nan
              
            #Path_LongestWordLength
            word_filter_path = list(filter(str.isalpha, tokens_path))
            if not word_filter_path: 
              Path_LongestWordLength = 0
            else: 
              Path_LongestWordLength = len(max(word_filter_path, key=len))
              
            #LongestPathTokenLength
            if tokens_path:
              LongestPathTokenLength = len(max(tokens_path, key=len))
            else:
              LongestPathTokenLength = 0
              
            #delimeter_path 
            delimeter_path = (len(re.split(r'\W+', path))-1)
            
            #Query_LetterCount
            Query_LetterCount = 0
            for char in query: 
              if char.isalpha():
                Query_LetterCount += 1
                
            #Querylength
            Querylength = len(query)
            
            #Query_DigitCount
            Query_DigitCount = 0
            for char in query: 
              if char.isnumeric():
                Query_DigitCount += 1
                
            #SymbolCount_URL
            SymbolCount_URL = 0 
            for char in url:
              if char.isalnum() == False:
                SymbolCount_URL += 1
                
            #SymbolCount_Domain
            SymbolCount_Domain = 0 
            for char in domain:
              if char.isalnum() == False:
                SymbolCount_Domain += 1
                
            #SymbolCount_FileName
            SymbolCount_FileName = 0 
            for char in filename:
              if char.isalnum() == False:
                SymbolCount_FileName += 1
                
            #SymbolCount_Afterpath
            SymbolCount_Afterpath = 0
            for char in afterpath:
              if char.isalnum() == False:
                SymbolCount_Afterpath += 1
                
            #SymbolCount_Directoryname
            SymbolCount_Directoryname = 0
            for char in directory:
              if char.isalnum() == False:
                SymbolCount_Directoryname += 1
                
            #urlLen
            urlLen = len(url)
            
            #URL_Letter_Count
            URL_Letter_Count = 0
            for letter in url: 
              if letter.isalpha():
                URL_Letter_Count += 1
                
            #URL_DigitCount
            URL_DigitCount = 0
            for char in url: 
              if char.isnumeric():
                URL_DigitCount += 1
                
            #delimeter_url
            delimeter_url = (len(re.split(r'\W+', url))-1)
            
            #vowel_ratio_url
            num_vowels = 0
            for char in url:
              if char in "aeiouAEIOU":
                num_vowels += 1 
            if len(url) != 0:
              vowel_ratio_url = num_vowels / len(url)
            else:
              vowel_ration_url = np.nan
        
            df.loc[len(df.index)] = [Arguments_LongestWordLength, argDomanRatio, ArgUrlRatio, argPathRatio, ArgLen, 
                                 avgdomaintokenlen, longdomaintokenlen, tld, delimeter_Domain, domainlength,
                                 domain_token_count, domain_digit_count, domain_letter_count, domainUrlRatio, Domain_LongestWordLength,
                                 Directory_DigitCount, Directory_LongestWordLength, Directory_LetterCount, Entropy_Afterpath, 
                                 Entropy_Domain, Entropy_DirectoryName, Entropy_URL, Entropy_Filename, 
                                 File_name_DigitCount, Filename_LetterCount, NumberofDotsinURL, NumberRate_Domain, 
                                 NumberRate_FileName, NumberRate_URL, NumberRate_DirectoryName, avgpathtokenlen, pathLength, 
                                 path_token_count, pathDomainRatio, pathurlRatio, Path_LongestWordLength, LongestPathTokenLength, 
                                 delimeter_path, Query_LetterCount, Querylength, Query_DigitCount, SymbolCount_URL,
                                 SymbolCount_Domain, SymbolCount_FileName, SymbolCount_Afterpath, SymbolCount_Directoryname, 
                                 urlLen, URL_Letter_Count, URL_DigitCount, delimeter_url, vowel_ratio_url]
            
                    
            df = scaler.transform(df)
            
            
            Klasse = FinalModel.predict(df)
            
            Probability = FinalModel.predict_proba(df)
            
            st.success("Your URL has been classified")
            
            for x in [Klasse]:
                        if x == 0:
                               st.success("Es handelt sich mit %i Prozent Sicherheit um einen unschädlichen URL!" % (Probability[:,0]*100))
                        if x == 1:
                               st.error("Es handelt sich zu %i Prozent um Malware!" % (Probability[:,1]*100))
                               st.write("Was ist Malware?")
                               st.caption("Malware ist ein Oberbegriff für verschiedene Formen von Schadsoftware, welche durch das Aufrufen des Links auf einem Gerät installiert werden. Die Installation erfolgt meist unbemerkt, arbeitet unauffällig im Hintergrund weiter und kann je länger die Infektion unentdeckt bleibt weitreichende Konsequenzen haben, wobei nicht nur PCs, sondern auch Tablets oder Smartphones betroffen sein können. Die bekanntesten Ausprägungen von Malware sind Trojaner oder Viren, weitere können den Betroffenen im schlimmsten Fall den Zugriff auf ihre Geräte gegen Lösegeld verweigern, durch das Sammeln von Informationen Identitätsdiebstahl veranlassen oder sich auf weitere Geräte ausbreiten. ")
                               st.write("Wie kann ich wissen, ob meine Geräte infiltriert wurden und wie gehe ich dagegen vor?")
                               st.caption("Dass das Gerät auffällig langsam operiert, sich der Speicher rasant füllt oder unerwartet Fenster geöffnet werden, können Indikatoren für Malware sein. Da dies aber auch gewöhnliche Symptome einer alternden Maschine sind, sollten sie nicht prinzipiell ein Grund zur Beunruhigung sein. Es kann dennoch auch präventiv sinnvoll sein, eine Antivirensoftware zum regelmässigen Scannen und in den meisten Fällen anschliessendem Entfernen von Malware zu installieren. Zwar tritt beispielsweise iOS Malware selten auf, dennoch kann Anti-Malware-Schutz nie zu 100 % gewährleistet werden. Oft handelt es sich bei Cybersicherheitssoftware um kostenpflichtige Angebote, wobei Nutzen-Kosten bzw. Schadensabwägungen getroffen werden sollten.")
                               st.caption("Quelle: https://www.avast.com/de-de/c-malware#topic-5")
                               
                        if x == 2:
                                st.error("Es handelt sich mit %i Prozent Sicherheit um Defacement!" % (Probability[:,2]*100))
                                st.write("Was ist Defacement?")
                                st.caption("Als eine der bekanntesten Angriffstechniken versuchen Hacker und Hackerinnen beim Defacement, Websites zu verfälschen und Organisationen zu schaden. Die übersetzte „Verunstaltung“ zielt häufig auf die Veränderung des Inhalts und des Erscheinungsbildes einer gesamten Website, wobei die böswillige Partei häufig die Reichweite einer Homepage zum Kundtun einer politischen oder religiösen Botschaft ausnutzen möchte.")
                                st.write("Wie bin ich von Defacement betroffen?")
                                st.caption("Aus Konsumentenperspektive gibt es kaum Massnahmen zur Präventation des Aufrufens einer defaced Website. Schliesslich könnte selbst Google wie 2012 in Rumänien zum Opfer der Methode werden, was anschliessend zu einer internen Wartung der Website führt und die Kundschaft insofern nicht weiter beeinflusst. Riskanter werden fehlleitende Informationen auf offiziellen Homepages der Regierung oder zudem verborgene Malware. Es gilt wie auch bei den anderen Klassifizierungen von URLegit, wachsam zu bleiben und etwaige Fälle zu melden.")
                                st.caption("Quelle: https://www.imperva.com/learn/application-security/website-defacement-attack/")   
                                
                        if x == 3:
                                st.error("Es handelt sich mit %i Prozent Sicherheit um Phishing!" % (Probability[:,3]*100))
                                st.write("Was ist Phishing?")
                                st.caption("„Sie haben gewonnen“, „Es gab ein Problem mit Ihrer Rechnung“ oder eine dringliche Nachricht der Bank sind typische Beispiele für versuchtes Phishing. Im Zuge des Telekommunikationsbetrugs versucht ein vermeintlich vertrauenswürdiger Angreifer, den Nutzer zum Aufrufen eines Links oder zum Herunterladen einer Datei zu bewegen, woraufhin dieser zur Aufführung privater Informationen wie Kreditkartendetails überlisten werden soll. Mithilfe verschiedener Kanäle und Strategien kann die Methode je nach Kompetenzen und Motiven der Cyberkriminellen weitreichende Konsequenzen bis hin zu Identitätsdiebstahl haben, wobei auch Unternehmen zum Opfer von Spionage werden können. ")
                                st.write("Wie kann ich wissen, ob ich betroffen bin und wie gehe ich dagegen vor?")
                                st.caption("Cybersicherheitssoftware ist dazu in der Lage, über Scannen auf Phishing aufmerksam zu machen und Werbeblocker einzusetzen. Grundsätzlich kann bei Phishing aber auch die Prävention äusserst effektiv sein: Das regelmässige Ändern von Passwörtern und eine generelle Vorsicht und Skepsis im Umgang mit potenziellen Fällen kann in Schulungsprogrammen von Sicherheitsteams erlernt werden.")
                                st.caption("Quelle: https://www.avast.com/de-de/c-phishing")
                                
                        
            
            st.write("Disclaimer")                    
            st.caption("Wir möchten maximale Sicherheit und Vorsicht im Umgang mit Cybersicherheit fördern. Aus diesem Grund haben wir bei der Programmierung und Auswahl unserer Modelle stets die Präzision maximiert. Das heisst, dass wir Links eher als z.B. Malware einstufen, obwohl sie Benign sind, anstatt zu risikieren, dass ein als Benign klassifzierter Link eigentlich Malware ist. Nichtsdestotrotz übernehmen wir keine Haftung für allfällige Schäden, die aufgrund eines von URLegit falsch klassifizierten Links entstehen.")
      
                                   
            st.subheader("Weitere Informationen")
            with st.expander("Sehen Sie sich hier die einzelnen Elemente der URL an"):
                st.caption(f"url: {url}")
                st.caption(f"Domain: {domain}")
                st.caption(f"Top level domain: {tld}")
                st.caption(f"Subdomain: {subdomain}")
                st.caption(f"Path: {path}")
                st.caption(f"Filename: {filename}")
                st.caption(f"Directory: {directory}")
                st.caption(f"Query: {query}")
                st.caption(f"Arguments: {arguments}")
                st.caption(f"Afterpath: {afterpath}")
            
     
            
            with st.expander("Sehen Sie sich hier die Features an, die zur Klassifikation der URL verwendet wurden"):
                 df = pd.DataFrame(df, columns = ["Arguments_LongestWordLength", "argDomanRatio", "ArgUrlRatio", "argPathRatio", "ArgLen", 
                                              "avgdomaintokenlen", "longdomaintokenlen", "tld", "delimeter_Domain", "domainlength",
                                              "domain_token_count", "domain_digit_count", "domain_letter_count", "domainUrlRatio", "Domain_LongestWordLength",
                                              "Directory_DigitCount", "Directory_LongestWordLength", "Directory_LetterCount", "Entropy_Afterpath", 
                                              "Entropy_Domain", "Entropy_DirectoryName", "Entropy_URL", "Entropy_Filename", 
                                              "File_name_DigitCount", "Filename_LetterCount", "NumberofDotsinURL", "NumberRate_Domain", 
                                              "NumberRate_FileName", "NumberRate_URL", "NumberRate_DirectoryName", "avgpathtokenlen", "pathLength", 
                                              "path_token_count", "pathDomainRatio", "pathurlRatio", "Path_LongestWordLength", "LongestPathTokenLength", 
                                              "delimeter_path", "Query_LetterCount", "Querylength", "Query_DigitCount", "SymbolCount_URL",
                                              "SymbolCount_Domain", "SymbolCount_FileName", "SymbolCount_Afterpath", "SymbolCount_Directoryname", 
                                              "urlLen", "URL_Letter_Count", "URL_DigitCount", "delimeter_url", "vowel_ratio_url"] )
                 st.table(df)
            
            with st.expander("Sehen Sie sich hier die Feature Importance an"):
                Importance = np.array(FinalModel.feature_importances_)
                Features = np.array(["Arguments_LongestWordLength", "argDomanRatio", "ArgUrlRatio", "argPathRatio", "ArgLen", 
                                             "avgdomaintokenlen", "longdomaintokenlen", "tld", "delimeter_Domain", "domainlength",
                                             "domain_token_count", "domain_digit_count", "domain_letter_count", "domainUrlRatio", "Domain_LongestWordLength",
                                             "Directory_DigitCount", "Directory_LongestWordLength", "Directory_LetterCount", "Entropy_Afterpath", 
                                             "Entropy_Domain", "Entropy_DirectoryName", "Entropy_URL", "Entropy_Filename", 
                                             "File_name_DigitCount", "Filename_LetterCount", "NumberofDotsinURL", "NumberRate_Domain", 
                                             "NumberRate_FileName", "NumberRate_URL", "NumberRate_DirectoryName", "avgpathtokenlen", "pathLength", 
                                             "path_token_count", "pathDomainRatio", "pathurlRatio", "Path_LongestWordLength", "LongestPathTokenLength", 
                                             "delimeter_path", "Query_LetterCount", "Querylength", "Query_DigitCount", "SymbolCount_URL",
                                             "SymbolCount_Domain", "SymbolCount_FileName", "SymbolCount_Afterpath", "SymbolCount_Directoryname", 
                                             "urlLen", "URL_Letter_Count", "URL_DigitCount", "delimeter_url", "vowel_ratio_url"])
                
                Variable_Importance = pd.DataFrame({'Feature': Features, 'Importance': list(Importance)}, columns=['Feature', 'Importance'])
                Variable_Importance = Variable_Importance.nlargest(20, ["Importance"])
                fig, ax = plt.subplots()
                ax = sns.barplot(x= "Importance", y="Feature", data = Variable_Importance)
                ax.set_title("Das sind die zwanzig wichtigsten Features")
                st.pyplot(fig)
                

                
    
###### Feature extraction
st.header("Klassifizieren von mehreren URLs")
uploaded_data = st.file_uploader("Laden Sie eine .csv Datei hoch, um mehrere URLs auf einmal zu kategorisieren")

if uploaded_data is not None:
    
    urls = pd.read_csv(uploaded_data, header = None)
    with st.form("Klassifizierung"):
        submitted4 = st.form_submit_button("Klassifizieren") 
    
    if submitted4:
    
        df = pd.DataFrame(columns = ["Arguments_LongestWordLength", "argDomanRatio", "ArgUrlRatio", "argPathRatio", "ArgLen", 
                                 "avgdomaintokenlen", "longdomaintokenlen", "tld", "delimeter_Domain", "domainlength",
                                 "domain_token_count", "domain_digit_count", "domain_letter_count", "domainUrlRatio", "Domain_LongestWordLength",
                                 "Directory_DigitCount", "Directory_LongestWordLength", "Directory_LetterCount", "Entropy_Afterpath", 
                                 "Entropy_Domain", "Entropy_DirectoryName", "Entropy_URL", "Entropy_Filename", 
                                 "File_name_DigitCount", "Filename_LetterCount", "NumberofDotsinURL", "NumberRate_Domain", 
                                 "NumberRate_FileName", "NumberRate_URL", "NumberRate_DirectoryName", "avgpathtokenlen", "pathLength", 
                                 "path_token_count", "pathDomainRatio", "pathurlRatio", "Path_LongestWordLength", "LongestPathTokenLength", 
                                 "delimeter_path", "Query_LetterCount", "Querylength", "Query_DigitCount", "SymbolCount_URL",
                                 "SymbolCount_Domain", "SymbolCount_FileName", "SymbolCount_Afterpath", "SymbolCount_Directoryname", 
                                 "urlLen", "URL_Letter_Count", "URL_DigitCount", "delimeter_url", "vowel_ratio_url"])
    
        for url in np.ravel(urls):
    
        ############################################
        ###############  Features   ################
        ############################################
    
            #Domain
            domain = urlparse(url).netloc
            try:
              tld = get_tld(url, fix_protocol=True)
            except: 
              tld = str()
            subdomain = tldextract.extract(url).subdomain
            #Path
            path = urlparse(url).path
            filename = os.path.basename(path)
            directory = os.path.dirname(path)
            query = urlparse(url).query
            arguments = query
            afterpath = urlparse(url).params + urlparse(url).query + urlparse(url).fragment
            #Tokenizer and Tokens
            tokenizer = RegexpTokenizer(r'\w+')
            #Arguments_LongestWordLength
            tokens_arguments = tokenizer.tokenize(arguments)
            word_filter_arguments = list(filter(str.isalpha, tokens_arguments))
            if not word_filter_arguments: 
              Arguments_LongestWordLength = 0
            else: 
              Arguments_LongestWordLength = len(max(word_filter_arguments, key=len))
            #argDomanRatio
            if len(domain) != 0:
              argDomanRatio = len(arguments)/len(domain)
            else: 
              argDomanRatio = np.nan
            #ArgUrlRatio
            if len(url) != 0:
              ArgUrlRatio = len(arguments)/len(url)
            else: 
              ArgUrlRatio = np.nan
            #argPathRatio
            if len(path) != 0:
              argPathRatio = len(arguments)/len(path)
            else: 
              argPathRatio = np.nan
            #ArgLen
            ArgLen = len(arguments)
            #avgdomaintokenlen
            tokens_domain = tokenizer.tokenize(domain)
            tokens_length = 0
            for token in tokens_domain: 
              tokens_length += len(token)
            if len(tokens_domain) != 0:
              avgdomaintokenlen = tokens_length / len(tokens_domain)
            else:
              avgdomaintokenlen = np.nan
            #longdomaintokenlen
            if not tokens_domain:
              longdomaintokenlen = 0
            else:
              longdomaintokenlen = len(max(tokens_domain, key=len))
            #tld 
            tld = len(tld)
            #delimeter_Domain #Frage ob da der Punkt auch mit reinzählt- in meiner Lösung ja
            delimeter_Domain = (len(re.split(r'\W+', domain))-1)
            #domainlength
            domainlength = len(domain)
            #domain_token_count
            domain_token_count = len(tokens_domain) 
            #domain_digit_count
            domain_digit_count = 0
            for char in domain: 
              if char.isnumeric():
                domain_digit_count += 1
            #domain_letter_count (Host = Domain)
            domain_letter_count = 0
            for letter in domain: 
              if letter.isalpha():
                domain_letter_count += 1
            #domainUrlRatio
            if len(url) != 0:
              domainUrlRatio = len(domain)/len(url)
            else:
              domainUrlRatio = np.nan
            #Domain_LongestWordLength
            word_filter_domain = list(filter(str.isalpha, tokens_domain))
            if not word_filter_domain: 
              Domain_LongestWordLength = 0
            else: 
              Domain_LongestWordLength = len(max(word_filter_domain, key=len))
            #Directory_DigitCount
            Directory_DigitCount = 0
            for char in directory: 
              if char.isnumeric():
                Directory_DigitCount += 1
            #Directory_LongestWordLength 
            tokens_directory = tokenizer.tokenize(directory)
            word_filter_directory = list(filter(str.isalpha, tokens_directory))
            if not word_filter_directory: 
              Directory_LongestWordLength = 0
            else: 
              Directory_LongestWordLength = len(max(word_filter_directory, key=len))
            #Directory_LetterCount
            Directory_LetterCount = 0
            for letter in directory:
              if letter.isalpha():
                Directory_LetterCount += 1
            
            #Entropy_Afterpath
            Entropy_Afterpath = entropy(afterpath)
            #Entropy_Domain
            Entropy_Domain =  entropy(domain)
            #Entropy_DirectoryName
            Entropy_DirectoryName = entropy(directory)
            #Entropy_URL
            Entropy_URL = entropy(url)
            #Entropy_Filename
            Entropy_Filename = entropy(filename)
            
            #File_name_DigitCount
            File_name_DigitCount = 0
            for char in filename: 
              if char.isnumeric():
                File_name_DigitCount += 1
            #Filename_LetterCount
            Filename_LetterCount = 0
            for letter in filename:
              if letter.isalpha():
                Filename_LetterCount += 1
            #NumberofDotsinURL
            NumberofDotsinURL = url.count(".")
            #NumberRate_Domain
            number_domain = 0 
            for char in domain:
              if char.isdigit():
                number_domain += 1
            if len(domain) != 0:
              NumberRate_Domain = number_domain / len(domain)
            else: 
              NumberRate_Domain = np.nan
            #NumberRate_FileName
            Number_filename = 0 
            for char in filename:
              if char.isdigit():
                Number_filename += 1
            if len(filename) != 0:
              NumberRate_FileName = Number_filename / len(filename)
            else:
              NumberRate_FileName = np.nan
            #NumberRate_URL
            number_url = 0
            for char in url:
              if char.isdigit():
                number_url += 1
            if len(url) != 0:
              NumberRate_URL = number_url / len(url)
            else: 
              NumberRate_URL = np.nan 
            #NumberRate_DirectoryName
            number_directory = 0 
            for char in directory:
              if char.isdigit():
                number_directory += 1
            if len(directory) != 0:
              NumberRate_DirectoryName = number_directory / len(directory)
            else:
              NumberRate_DirectoryName = np.nan
            #avgpathtokenlen
            tokens_path = tokenizer.tokenize(path)
            tokens_length = 0
            for token in tokens_path: 
              tokens_length += len(token)
            if len(tokens_path) != 0:
              avgpathtokenlen = tokens_length / len(tokens_path)
            else:
              avgpathtokenlen = np.nan
            #pathLength
            pathLength = len(path)
            #path_token_count
            path_token_count = len(tokens_path)
            #pathDomainRatio
            if len(domain) != 0:
              pathDomainRatio = len(path)/len(domain)
            else:
              pathDomainRatio = np.nan
            #pathurlRatio
            if len(url) != 0:
              pathurlRatio = len(path)/len(url)
            else: 
              pathurlRatio = np.nan
            #Path_LongestWordLength
            word_filter_path = list(filter(str.isalpha, tokens_path))
            if not word_filter_path: 
              Path_LongestWordLength = 0
            else: 
              Path_LongestWordLength = len(max(word_filter_path, key=len))
            #LongestPathTokenLength
            if tokens_path:
              LongestPathTokenLength = len(max(tokens_path, key=len))
            else:
              LongestPathTokenLength = 0
            #delimeter_path 
            delimeter_path = (len(re.split(r'\W+', path))-1)
            #Query_LetterCount
            Query_LetterCount = 0
            for char in query: 
              if char.isalpha():
                Query_LetterCount += 1
            #Querylength
            Querylength = len(query)
            #Query_DigitCount
            Query_DigitCount = 0
            for char in query: 
              if char.isnumeric():
                Query_DigitCount += 1
            #SymbolCount_URL
            SymbolCount_URL = 0 
            for char in url:
              if char.isalnum() == False:
                SymbolCount_URL += 1
            #SymbolCount_Domain
            SymbolCount_Domain = 0 
            for char in domain:
              if char.isalnum() == False:
                SymbolCount_Domain += 1
            #SymbolCount_FileName
            SymbolCount_FileName = 0 
            for char in filename:
              if char.isalnum() == False:
                SymbolCount_FileName += 1
            #SymbolCount_Afterpath
            SymbolCount_Afterpath = 0
            for char in afterpath:
              if char.isalnum() == False:
                SymbolCount_Afterpath += 1
            #SymbolCount_Directoryname
            SymbolCount_Directoryname = 0
            for char in directory:
              if char.isalnum() == False:
                SymbolCount_Directoryname += 1
            #urlLen
            urlLen = len(url)
            #URL_Letter_Count
            URL_Letter_Count = 0
            for letter in url: 
              if letter.isalpha():
                URL_Letter_Count += 1
            #URL_DigitCount
            URL_DigitCount = 0
            for char in url: 
              if char.isnumeric():
                URL_DigitCount += 1
            #delimeter_url
            delimeter_url = (len(re.split(r'\W+', url))-1)
            #vowel_ratio_url
            num_vowels = 0
            for char in url:
              if char in "aeiouAEIOU":
                num_vowels += 1 
            if len(url) != 0:
              vowel_ratio_url = num_vowels / len(url)
            else:
              vowel_ration_url = np.nan
        
            df.loc[len(df.index)] = [Arguments_LongestWordLength, argDomanRatio, ArgUrlRatio, argPathRatio, ArgLen, 
                                     avgdomaintokenlen, longdomaintokenlen, tld, delimeter_Domain, domainlength,
                                     domain_token_count, domain_digit_count, domain_letter_count, domainUrlRatio, Domain_LongestWordLength,
                                     Directory_DigitCount, Directory_LongestWordLength, Directory_LetterCount, Entropy_Afterpath, 
                                     Entropy_Domain, Entropy_DirectoryName, Entropy_URL, Entropy_Filename, 
                                     File_name_DigitCount, Filename_LetterCount, NumberofDotsinURL, NumberRate_Domain, 
                                     NumberRate_FileName, NumberRate_URL, NumberRate_DirectoryName, avgpathtokenlen, pathLength, 
                                     path_token_count, pathDomainRatio, pathurlRatio, Path_LongestWordLength, LongestPathTokenLength, 
                                     delimeter_path, Query_LetterCount, Querylength, Query_DigitCount, SymbolCount_URL,
                                     SymbolCount_Domain, SymbolCount_FileName, SymbolCount_Afterpath, SymbolCount_Directoryname, 
                                     urlLen, URL_Letter_Count, URL_DigitCount, delimeter_url, vowel_ratio_url]
    try:
        df = scaler.transform(df)
        
    except:
        st.stop()
    
          
    Predictions = FinalModel.predict(df)
     
    bins = [0,1,2,3,4]
    names = ["Benign", "Malware", "Defacement", "Phishing", "Spam"]
    d = dict(enumerate(names, 1))
    
    urls["Kategorie"] = np.vectorize(d.get)(np.digitize(Predictions, bins))
    
    # Add User Feedback
    st.success("👍 Es wurden erfolgreich %i URLs klassifiziert!👍" % urls.shape[0])
    
    # Add Download Button
    st.download_button(label = "Klassifizierte URLs herunterladen",
                       data = urls.to_csv(header = None, index = False).encode("utf-8"),
                       file_name = "predicted_urls.csv")
    
    
st.header("Visualisierung")

with st.expander("Öffnen um zu den Visualisierungen zu gelangen"):
           
       bins = [0,1,2,3,4]
       names = ["Benign", "Malware", "Defacement", "Phishing", "Spam"]
       d = dict(enumerate(names, 1))
       
       Data["Kategorie"] = np.vectorize(d.get)(np.digitize(Data["label"], bins))
       
       st.write("Vergleichen Sie die Durchschnittswerte der Features in den verschiedenen Kategorien")
       with st.form("Visualisierung"):
           names = Data.columns.drop(["label", "Kategorie"])
           variable = st.selectbox("Variable zum vergleichen auswählen", names)
           submitted3 = st.form_submit_button("Visualisieren")             

       if submitted3:                 
           barplotdata = Data[["Kategorie", variable]].groupby("Kategorie").mean()
           fig1, ax = plt.subplots(figsize=(8,3.7))
           ax.bar(barplotdata.index.astype(str), barplotdata[variable], color = "#fc8d62")
           ax.set_ylabel(variable)
       
           st.pyplot(fig1, use_container_width= False) 
       
       st.write("Schauen Sie sich die Korrelationen der Features an")       
       with st.form("Visualisierung2"):
           Variable1 = st.selectbox("Variable 1", names)
           Variable2 = st.selectbox("Variable 2", names)
           submitted4 = st.form_submit_button("Visualisieren")    
       
       if submitted4:
          fig2, ax = plt.subplots(figsize=(8,3.7))
          plt.scatter(Data[Variable1], Data[Variable2], color = "black")
          ax.set_ylabel(Variable2)
          ax.set_xlabel(Variable1)
          ax.set_title(f"Korrelation zwischen {Variable1} und {Variable2}")
          m, b = np.polyfit(Data[Variable1], Data[Variable2], 1)
          plt.plot(Data[Variable1], m*Data[Variable1]+b, color = "red")
          st.pyplot(fig2, use_container_width = False)
    
    
st.markdown("Business Analytics und Data Science Applications, Universität St. Gallen")
st.caption("Johannes Strigl, Tom Salzsieder, Samuel Sattler, Carina Bund, Jonas Kleubler")
