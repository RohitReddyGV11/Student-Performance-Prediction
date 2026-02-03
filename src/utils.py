import os
import sys
import dill
import pandas as pd
import numpy as np

def save_obj(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)