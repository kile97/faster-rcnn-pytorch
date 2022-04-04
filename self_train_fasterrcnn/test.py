import numpy as np

root = r'/data/kile/other/yolov3/data_set_kile/data_txt/enhance_train_train.txt'

def cleanData():
    realdata = []
    with open(root, "r") as rdata:
        contents = rdata.readlines()
    for i in contents:
        if len(i.split(" ")) <= 1:
            print(f"no box :{i}")
            continue
        boxes = i.replace("\n","").split(" ")
        str_ = boxes[0]
        boxes = boxes[1:]
        for box in boxes:
            xmin, ymin, xmax, ymax = np.int(box.split(",")[0]), np.int(box.split(",")[1]), np.int(box.split(",")[2]), np.int(box.split(",")[3])
            if xmin < xmax and ymin < ymax:
                str_ += " "
                str_ += box
        if len(str_.split(" ")) > 1:
            realdata.append(str_)
    with open(r"/data/kile/other/yolov3/data_set_kile/data_txt/enhance_train_train1.txt", "w") as w:
        for data in realdata:
            w.write(data)
            w.write("\n")

cleanData()