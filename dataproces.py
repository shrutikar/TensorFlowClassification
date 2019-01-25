import json
import os
import re

def genClasses(filename):
    with open(filename,'r') as file :
        lines = file.readlines()
    classes = list()
    for line in lines:
        if line[-3:-1]=='ch' : cl = 1
        if line[-3:-1]=='hi' : cl = 2
        if line[-3:-1]=='bi' : cl = 3
        if line[-3:-1]=='dt' : cl = 4
        if line[-3:-1]=='ph' : cl = 5
        if line[-3:-1]=='ar' : cl = 6
        if line[-3:-1]=='ss' : cl = 7
        if line[-3:-1]=='ge' : cl = 8
        if line[-3:-1]=='ma' : cl = 9
        if line[-3:-1]=='ot' : cl = 10
        if line[-3:-1]=='en' : cl = 10
        classes.append(cl)
                
    return classes

def readJsons(directory,classes=[]):
    file_json = [x for x in os.listdir(directory) if x.endswith(".json")]
    js_out = list()
    lecture_id = 0
    print("Reading all json files from {}".format(directory))
    for filename in file_json :
        print("{} : {}".format(lecture_id,filename))

        with open(directory+"\\"+filename) as file:
            js_file = json.load(file)
        for slide in js_file["slides"] :
            slide_dict = dict()

            #creating audio_alternatives and audio_confidence
            audio_alternatives = list()
            first = True
            audio_confidence = 0.0;
            for result in slide["audio"] :
                if "confidence" in result["result"][0]["alternative"][0]:
                    audio_confidence += result["result"][0]["alternative"][0]["confidence"]
                else:
                    audio_confidence += 0.5
                if first :
                    first = False
                    for i in range(5) :
                        if i < len(result["result"][0]["alternative"]) :
                            audio_alternatives.append(result["result"][0]["alternative"][i]["transcript"])
                        else :
                            audio_alternatives.append('')

                else:
                    for i in range(len(result["result"][0]["alternative"])) :
                        audio_alternatives[i] += result["result"][0]["alternative"][i]["transcript"]
            slide_dict["audio_alternatives"] = audio_alternatives
            if len(slide["audio"])>0 :
                audio_confidence /= len(slide["audio"])
            slide_dict["audio_confidence"] = audio_confidence

            #creating audio_stem and audio_stem_alter
            audio_stem = ""
            audio_stem_alter = ""
            for i in range(len(slide["audio_stemm"])):
                 audio_stem += slide["audio_stemm"][i]
                 audio_stem_alter += slide["audio_stemm_alter"][i]
            slide_dict["audio_stem"] = audio_stem
            slide_dict["audio_stem_alter"] = audio_stem_alter

            slide_dict["ocr_stem"] = slide["ocr_stemm"]
            slide_dict["ocr"] = slide["ocr"]
            slide_dict["duration"] = slide["end"]-slide["start"]

            slide_dict["lecture_id"] = lecture_id

            if len(classes) != 0 :
                slide_dict["label"] = classes[lecture_id]
            for a in slide_dict  : 
                if type(slide_dict[a]) is str :
                    slide_dict[a] = re.sub("\\n"," ",slide_dict[a])
            js_out.append(slide_dict)
        lecture_id += 1;
    return js_out

def writeJson(directoryOut,directoryJson,classes):
    with open(directoryOut + '\\data.json', 'w') as outfile:
        json.dump(readJsons(directoryJson,classes), outfile)

