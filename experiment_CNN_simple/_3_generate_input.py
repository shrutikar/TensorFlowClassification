import dataop as dop

a = dop.load_var("audio_stem_join")

d = dop.load_var("dicts/audio_stem_all_dict")
d = d[0]

#34910
inp = dop.create_indexMatrix(a,d,5000)
dop.save_var(inp,"inputs/audio_stem")


#a = dop.load_var("audio_join")

#d = dop.load_var("dicts/audio_alternatives_dict")
#d = d[0]

#inp = dop.create_indexMatrix(a,d,5)
#dop.save_var(inp,"inputs/audio")


