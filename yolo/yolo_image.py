import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import sys
import cv2
import os


def trim(img, box, outdir, mode="extend"):
    """
    trim image file with result of yolo detection 

    Parameters
    ----------
    img : str
        input image file path
    box : numpy array
        result of yolo detection
    outdir : str
        output directory path
    mode : str
        box trim strategy
        fit:original size
        extend:bigger size
    """

    os.makedirs(outdir,exist_ok=True)
    image = cv2.imread(img) 
    name = os.path.basename(img)
    name = name.rsplit(".",1)[0]
    name = name.rsplit("_",1)[0]
    for i in range(box.shape[0]):
        if(mode == "fit"):
            trim = image[box[i][0]:box[i][1], box[i][2]:box[i][3]]
        if(mode == "extend"):
            new_box = [0]*4
            new_box[0],new_box[1],new_box[2],new_box[3] = int(box[i][0]/1.2),int(1.2*box[i][1]),int(box[i][2]/1.2),int(1.2*box[i][3])
            trim = image[new_box[0]:new_box[1], new_box[2]:new_box[3]]
        if box.shape[0]==1:
            outpath = os.path.join(outdir,name+".png")
        else:
            outpath = os.path.join(outdir,name+"-"+str(i)+".png")
        cv2.imwrite(outpath,trim)
