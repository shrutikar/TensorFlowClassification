import dataproces as dp
import json

cl = dp.genClasses("data\\titles_dec.txt")
dp.writeJson("data","data\\json",cl)
# time 14.25