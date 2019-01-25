import tensorflow as tf
import json
import collections
import numpy as np
from unidecode import unidecode
import re
import random
from math import floor 
import os
import warnings

folder = "data\\"

def saveAll(directory,audio_alternatives,audio_confidence,audio_stem,audio_stem_alter,duration,label,lecture_id,ocr_stem,ocr) :
    save_var(audio_alternatives,directory+"audio_alternatives")
    save_var(audio_confidence,directory+"audio_confidence")
    save_var(audio_stem,directory+"audio_stem")
    save_var(audio_stem_alter,directory+"audio_stem_alter")
    save_var(duration,directory+"duration")
    save_var(label,directory+"label")
    save_var(lecture_id,directory+"lecture_id")
    save_var(ocr_stem,directory+"ocr_stem")
    save_var(ocr,directory+"ocr")

def saveSepar(datafilename) :
    with open(datafilename,'r') as file:
        data = json.load(file)
    audio_alternatives = list()
    audio_confidence = list()
    audio_stem = list()
    audio_stem_alter = list()
    audio_stem_all = list()
    ocr_stem = list()
    ocr = list()
    duration = list()
    lecture_id = list()
    label = list()
    for d in data:
        audio_alternatives.append(d["audio_alternatives"])
        audio_confidence.append(d["audio_confidence"])
        audio_stem.append(d["audio_stem"])
        audio_stem_alter.append(d["audio_stem_alter"])
        audio_stem_all.append(d["audio_stem"]+" "+d["audio_stem_alter"])
        ocr_stem.append(d["ocr_stem"])
        ocr.append(d["ocr"])
        duration.append(d["duration"])
        lecture_id.append(d["lecture_id"])
        label.append(d["label"])
    save_var(audio_stem_all,"audio_stem_all")
    saveAll("",audio_alternatives,audio_confidence,audio_stem,audio_stem_alter,duration,label,lecture_id,ocr_stem,ocr)

def save_var(var,filename) :
    with open(folder+filename+".json","w") as file:
        json.dump(var,file)

def save_np(var,filename):
    with open(folder+filename+".json","wb") as file:
        np.save(file,var)

def load_var(filename) :
    with open(folder+filename+".json","r") as file :
        return json.load(file)

def load_np(filename):
    with open(folder+filename+".json","rb") as file :
        return np.load(file)

def get_audio_alternatives():
    return load_var("audio_alternatives")

def get_allintext_audio_alternatives() :
    out = ""
    for alternatives in get_audio_alternatives():
        out += " "+" ".join(alternatives)
    return out

def get_audio_first_alternative():
    alternatives = load_var("audio_alternatives")
    out = list()
    for alternative in alternatives :
        print(alternative)
        if len(alternative)==0 :
            out.append(" ")
        else :
            out.append(alternative[0])
    print(out)
    input("nic")
    return out

def get_audio_other_alternatives() :
    alternatives = load_var("audio_alternatives")
    out = list()
    for alternative in alternatives :
        out.append("")
        first = True
        for text in alternative :
            if first :
                first = False
            else :
                out[-1] += text
    return out

    
def allintext(strlist):
    return " ".join(strlist)


def create_indexDicts(text, edit=True, char_level=False, char_n=3, smaller=True):
    if type(text) is list :
        if type(text[0]) is list :
            for i in range(len(text)):
                text[i] = allintext(text[i])
        text = allintext(text)
    if edit :
        text = edit_text(text)
    dictionary = {'_BLANK_':0}
    occurance_dict = dict()
    if char_level :
        if smaller :
            N=1
        else :
            N=char_n
        for n in range(N,char_n+1) :
            for i in range(len(text)-n+1) :
                if not text[i:i+n] in dictionary :
                    dictionary[text[i:i+n]] = len(dictionary)       
                    occurance_dict[text[i:i+n]] = 1
                else :
                    occurance_dict[text[i:i+n]] += 1     
    else :
        count = collections.Counter(re.split("\s+",text)).most_common()
        for word, occ in count:
            dictionary[word] = len(dictionary)
            occurance_dict[word] = occ
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary, occurance_dict

def create_dictsFromVar(var_load,char_level=False, char_n=3, smaller=True, varbose=False):
    d = create_indexDicts(load_var(var_load),char_level=char_level, char_n=char_n, smaller=smaller)
    if char_level:
        save_var(d,"dicts\\"+var_load+"_"+str(char_n)+"_dict")
    else:
        save_var(d,"dicts\\"+var_load+"_dict")
    if varbose:
        print("Dictionary of {} has size {}".format(var_load,len(d[0])))

def create_indexMatrix(strlist,indexDict,maxWords=0,char_level=False):
    if char_level:
        out = list()
        for text in strlist:
            out.append(list())
            for c in text:
                out[-1].append(indexDict[c])
        return out
    else:
        if maxWords==0:
            maxWords=len(indexDict)
        out = list()
        for text in strlist:
            out.append(list())
            sentence = re.split("\s+",text)
            for word in sentence:
                out[-1].append(min(indexDict[word],maxWords))
        return out

def edit_text(strlist, delNum=False):
    isstr = isinstance(strlist,str)
    if isstr :
        strlist = [strlist]
    out = list()
    for text in strlist:
        new = unidecode(text).lower()
        new = re.sub("[^a-z0-9]"," ",new)
        if delNum :
            new = re.sub("\d","",new)
        else :
            new = re.sub("\d","*",new) #substitute number by character *
        new = re.sub("\s+"," ",new)
        out.append(new)
    if isstr :
        return out[0]
    else :
        return out


def create_freq_input(strlist,indexDict,size=0, char_level=False, char_n=3):
    if size==0:
        size = len(indexDict)
    
    out = np.zeros([len(strlist),size+1],int)
    if char_level :
        char_n = len(max(indexDict.keys(), key=len))
        if len(min(indexDict.keys(), key=len))<char_n :
            N = 1
        else :
            N = char_n
        for i in range(len(strlist)) :
            for n in range(N,char_n) :
                for j in range(len(strlist[i])-n+1) :
                    if strlist[i][j:j+n] in indexDict :
                        out[i,min(indexDict[strlist[i][j:j+n]],size)]+=1
                    else :
                        out[i,size]+=1 #considered as Rare
    else :
        for i in range(len(strlist)) :
            for word in re.split("\s+",strlist[i]):
                if word in indexDict :
                    out[i,min(indexDict[word],size)]+=1
                else :
                    out[i,size]+=1 #considered as Rare
    return out

def get_inv_doc_freq(freqInput) :
    doc_freq = np.sum(freqInput!=0,axis=0)
    warnings.filterwarnings("ignore")
    inv_doc_freq = np.log(len(freqInput)/doc_freq)
    inv_doc_freq[doc_freq==0]=0
    warnings.resetwarnings()
    return inv_doc_freq

def tfidf_from_freq(freqInput,inv_doc_freq):
    max_freq = np.max(freqInput,axis=1)
    warnings.filterwarnings("ignore")
    term_frac = 1/max_freq
    term_frac[max_freq==0]=0
    warnings.resetwarnings()
    
    inv_doc_freq,term_frac = np.meshgrid(inv_doc_freq,term_frac)
    
    return np.multiply(inv_doc_freq,np.multiply(freqInput,term_frac))

def tfidf(freqInput):
    return tfidf_from_freq(freqInput,get_inv_doc_freq(freqInput))

def del_features(tr_data,val_data,min_observations=1) :
    oc_fe = np.sum(tr_data!=0,axis=0) #occurence of features
    mask = oc_fe>=min_observations
    return tr_data[:,mask],val_data[:,mask]


"""
def create_crossvalSets(saved_data_var="data",K=10) :
    # also edited text variables
    data = load_var(saved_data_var)
    i = 0
    for element in data :
        element["slide_id"] = i
        i+=1
    random.shuffle(data)
    L = len(data)
    try:
        os.stat(folder+str(K)+"-crossval")
    except:
        os.mkdir(folder+str(K)+"-crossval")  
    for k in range(K):
        audio_alternatives = list()
        audio = list()
        audio_confidence = list()
        audio_stem = list()
        audio_stem_alter = list()
        duration = list()
        label = list()
        lecture_id = list()
        ocr_stem = list()
        ocr = list()
        slide_id = list()
        for i in range(floor(k*L/K),floor((k+1)*L/K)) :
            audio_alternatives.append(edit_text(data[i]["audio_alternatives"]))
            if edit_text(data[i]["audio_alternatives"]) :
                audio.append(edit_text(data[i]["audio_alternatives"][0]))
            else:
                audio.append("")
            audio_confidence.append(data[i]["audio_confidence"])
            audio_stem.append(edit_text(data[i]["audio_stem"]))
            audio_stem_alter.append(edit_text(data[i]["audio_stem_alter"]))
            duration.append(data[i]["duration"])
            label.append(data[i]["label"])
            lecture_id.append(data[i]["lecture_id"])
            #ocr_stem.append(edit_text(data[i]["ocr_stem"]))
            ocr.append(edit_text(data[i]["ocr"]))
            slide_id.append(data[i]["slide_id"])
        directory = str(K)+"-crossval\\"+str(k)+"\\"
        try:
            os.stat(folder+directory)
        except:
            os.mkdir(folder+directory)  
        saveAll(directory,audio_alternatives,audio_confidence,audio_stem,audio_stem_alter,duration,label,lecture_id,ocr_stem,ocr)
        save_var(slide_id,directory+"slide_id")
        save_var(audio,directory+"audio")

        label_counts = dict()
        for l,f in collections.Counter(label).most_common() :
            label_counts[str(l)] = f
        save_var(label_counts,directory+"label_counts")

        ne = list()
        ne_label = list()
        ne_id = list()
        for i in range(len(audio_alternatives)) :
            if  audio_alternatives[i] and audio_alternatives[i][0] :
                ne.append(audio_alternatives[i][0])
                ne_label.append(label[i])
                ne_id.append(slide_id[i])
        save_var(ne,directory+"ne_audio")
        save_var(ne_label,directory+"ne_audio_label")
        save_var(ne_id,directory+"ne_audio_id")
        label_counts = dict()
        for l,f in collections.Counter(ne_label).most_common() :
            label_counts[str(l)] = f
        save_var(label_counts,directory+"ne_audio_label_counts")
                
        def create_ne(var, name) :
            ne = list()
            ne_label = list()
            ne_id = list()
            for i in range(len(var)) :
                if var[i] : # if is not an empty string
                    ne.append(var[i])
                    ne_label.append(label[i])
                    ne_id.append(slide_id[i])
            save_var(ne,directory+"ne_"+name)
            save_var(ne_label,directory+"ne_"+name+"_label")
            save_var(ne_id,directory+"ne_"+name+"_id")
            label_counts = dict()
            for l,f in collections.Counter(ne_label).most_common() :
                label_counts[str(l)] = f
            save_var(label_counts,directory+"ne_"+name+"_label_counts")
        
        create_ne(audio_stem,"audio_stem")
        create_ne(ocr,"ocr")

def create_crossvalAllDicts(K=10) :
    
    def create_crossvalDict(name,K=10,char_level=False) :
        for k in range(K):
            text = list()
            for i in range(K):
                if i != k :
                    text = text + load_var(str(K)+"-crossval\\"+str(i)+"\\"+name)
            if char_level :
                save_var(create_indexDicts(allintext(text),char_level=True),str(K)+"-crossval\\"+str(k)+"\\oth_dict_"+name)
            else:
                save_var(create_indexDicts(allintext(text)),str(K)+"-crossval\\"+str(k)+"\\oth_dict_"+name)
    
    create_crossvalDict("ne_audio",K)
    create_crossvalDict("ne_ocr",K,char_level=True)
    create_crossvalDict("ne_audio_stem",K)
"""

def word_count(var_name) :
    text = load_var(var_name)
    if type(text) is list :
        if type(text[0]) is list :
            for i in range(len(text)):
                text[i] = allintext(text[i])
        text = allintext(text)
    text = edit_text(text)
    return len(text.split())

def create_crossvalAlldatasetsTFIDF(K=10):
    directory = str(K)+"-crossval\\"
    all_l = load_var("label")
    for k in range(K) :
        d,_ = load_var(directory+str(k)+"\\oth_dict_ne_audio_stem")
        v = create_freqInput(load_var(directory+str(k)+"\\ne_audio_stem"),d)
        #training
        tr_l = list()
        first = True
        for i in range(K):
            if i!=k :
                if first :
                    tr = create_freqInput(load_var(directory+str(i)+"\\ne_audio_stem"),d)
                    first=False
                else :
                    tr = np.concatenate((create_freqInput(load_var(directory+str(i)+"\\ne_audio_stem"),d),tr))
                for id in load_var(directory+str(i)+"\\ne_audio_stem_id") :
                    tr_l.append(all_l[id])

        idf = get_inv_doc_freq(tr)
        tr = tfidf_from_freq(tr,idf)
        #validation
        
        v_l = list()
        for id in load_var(directory+str(k)+"\\ne_audio_stem_id") :
            v_l.append(all_l[id])
        
        save_np(tfidf_from_freq(v,idf),directory+str(k)+"\\ne_audio_stem_val")
        save_np(np.array(v_l,dtype=np.int8),directory+str(k)+"\\ne_audio_stem_l_val")
        save_np(np.array(tr),directory+str(i)+"\\ne_audio_stem_tr")
        save_np(np.array(tr_l,dtype=np.int8),directory+str(i)+"\\ne_audio_stem_l_tr")
        a = imput()



def label2mat(labels) :
    out = np.zeros([len(labels),max(labels)])
    for i in range(len(labels)) :
        out[i][labels[i]-1]=1
    return out

def normalize_lin(tr_data,val_data):
    max_values = np.max(tr_data,0)
    min_values = np.min(tr_data,0)
    dif = max_values-min_values
    warnings.filterwarnings("ignore")
    for i in range(len(tr_data)):
        tr_data[i] = 2*(tr_data[i]-min_values)/dif-1
        tr_data[i][dif==0] = 0
    for i in range(len(val_data)):
        val_data[i] = 2*(val_data[i]-min_values)/dif-1
        val_data[i][dif==0] = 0
    warnings.resetwarnings()
    return tr_data,val_data



    
