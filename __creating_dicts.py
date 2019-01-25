import dataop

dataop.saveSepar("data/data.json")
dataop.create_dictsFromVar("audio_stem_all")
dataop.create_dictsFromVar("audio_join")
dataop.create_dictsFromVar("ocr",char_level=True,char_n=3)
dataop.create_dictsFromVar("ocr",char_level=True,char_n=1)
