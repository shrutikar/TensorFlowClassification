import dataop as dop

audio = dop.load_var("audio_alternatives")
new = list()
for slide in audio :
    txt = ""
    for alt in slide :
        txt += " " + alt
    new.append(dop.edit_text(txt))

new = dop.edit_text(new)
dop.save_var(new,"audio_join")


audio = dop.load_var("audio_stem_all")
new = list()
for slide in audio :
    txt = ""
    for alt in slide :
        txt += " " + alt
    new.append(dop.edit_text(txt))

new = dop.edit_text(new)
dop.save_var(new,"audio_stem_join")


