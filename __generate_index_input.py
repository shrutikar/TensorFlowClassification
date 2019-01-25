import dataop as dop

#--audio_stem--
a = dop.load_var("audio_stem_all")
a = dop.edit_text(a)

d = dop.load_var("dicts/audio_stem_all_dict")
d = d[0]

inp = dop.create_indexMatrix(a,d,50000)
dop.save_var(inp,"inputs/audio_stem")


#--ocr--
a = dop.load_var("ocr")
a = dop.edit_text(a)

d = dop.load_var("dicts/ocr_1_dict")
d = d[0]

inp = dop.create_indexMatrix(a,d,char_level=True)
dop.save_var(inp,"inputs/ocr_char")


#--audio_all--
a = dop.load_var("audio_join")
a = dop.edit_text(a)

d = dop.load_var("dicts/audio_join_dict")
d = d[0]

inp = dop.create_indexMatrix(a,d)
dop.save_var(inp,"inputs/audio_all")

