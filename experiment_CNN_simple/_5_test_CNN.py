from gtrain import gtrain
from data_model_CNN import Sentence_CNN
from data_model_CNN import DataForCNN
import dataop as dop


tr = dop.load_var("inputs/audio_stem_tr")
tr_l = dop.load_var("inputs/audio_stem_tr_l")
tst = dop.load_var("inputs/audio_stem_tst")
tst_l = dop.load_var("inputs/audio_stem_tst_l")

data = DataForCNN(tr,tr_l,tst,tst_l)
model = Sentence_CNN(10,5000)
gtrain(model,data,out_dir="runs\CNN",evaluate_every=1000,checkpoint_every=1000,num_epochs=10000)