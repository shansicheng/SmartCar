import configparser
# YOLOv5 Classes
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

def isfloat(tmp_str):
    str_list=tmp_str.split('.')
    if len(str_list)>2:
        return False
    else:
        for si in str_list:
            if not si.isdigit():
                return False
    return True
class Config:
    def __init__(self,load_file_name="./server.conf"):
        file_config = configparser.ConfigParser()
        file_config.read(load_file_name, encoding="utf-8")
        sections = file_config.sections()
        for sec_key in sections:
            for opt_key in file_config.options(sec_key):
                opt_value = file_config.get(sec_key,opt_key)
                if opt_value.isdigit():
                    opt_value=int(opt_value)
                elif isfloat(opt_value):
                    opt_value=float(opt_value)
                else:
                    pass
                setattr(self, opt_key, opt_value)
def get_config():
    config = Config()
    return config

    
            
