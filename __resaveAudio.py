import dataop as dop
import re

audio = dop.load_var("audio_alternatives")
new = list()
for slide in audio :
    txt = ""
    for alt in slide :
        txt += " " + alt
    new.append(re.sub("\d","",txt))

new = new
dop.save_var(new,"audio_join")


