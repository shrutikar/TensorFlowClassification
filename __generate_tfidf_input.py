import dataop as dop
import numpy as np


d = dop.load_var("dicts/audio_stem_all_dict")
a = np.array(list(d[2].values()))
occ2num = dict()
for u in set(a): occ2num[str(u)] = int(np.sum(a>=u))
dop.save_var(occ2num,"occ2num_audio_stem")

I = dop.tfidf(dop.create_freq_input(
    dop.load_var("audio_stem_all"),
    d[0],
    size=occ2num[str(15)]))
I,_ = dop.normalize_lin(I, [])
I = I[:,np.sum(I!=0,0)>0]

dop.save_np(I,"inputs/audio_stem_tfidf_norm")

#---ocr---

d = dop.load_var("dicts/ocr_3_dict")
a = np.array(list(d[2].values()))
occ2num = dict()
for u in set(a): occ2num[str(u)] = int(np.sum(a>=u))
dop.save_var(occ2num,"occ2num_ocr")

I = dop.tfidf(dop.create_freq_input(
    dop.load_var("audio_stem_all"),
    d[0],
    size=occ2num[str(1)],
    char_level=True,
    char_n=3))
I,_ = dop.normalize_lin(I, [])
I = I[:,np.sum(I!=0,0)>0]

dop.save_np(I,"inputs/ocr_tfidf_norm")
