import dataop as dop
import numpy as np
from math import floor

inp = np.array(dop.load_var("inputs/audio_stem"))
l = np.array(dop.load_var("label"))
num_classes = max(l)
p = floor(len(l)/10)

out = np.zeros([len(l),num_classes])
for i in range(len(l)) :
    out[i][l[i]-1]=1

rp = np.random.permutation(len(inp))
inp = inp[rp]
out = out[rp]


dop.save_var(rp.tolist(),"rp")
dop.save_var(inp[:p].tolist(),"inputs/audio_stem_tst")
dop.save_var(out[:p].tolist(),"inputs/audio_stem_tst_l")
dop.save_var(inp[p:].tolist(),"inputs/audio_stem_tr")
dop.save_var(out[p:].tolist(),"inputs/audio_stem_tr_l")

