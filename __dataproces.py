import dataproces as dp

cl = dp.genClasses("data\\titles_dec.txt")
dp.writeJson("data","data\\json",cl)
# time 14.25