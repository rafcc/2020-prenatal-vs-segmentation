#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import os, cv2
import statistics
import math

def load(folder_path):

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = []
    for i, image_file in enumerate(image_files):
        images.append(cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE))
    return images, image_files

def dice(y_true,y_pred):
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    if np.sum(y_t)==0 and np.sum(y_p)==0:
        dice = 0
    else:
        dice = np.sum(y_t[y_p == 1])*2.0 / (np.sum(y_t) + np.sum(y_p))   
    return dice


def IoU(y_true,y_pred):
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    if np.sum(y_t)==0 and np.sum(y_p)==0:
        IoU = 0
    else:
        IoU =  np.sum(y_t[y_p == 1])/ (np.sum(y_t) + np.sum(y_p) - np.sum(y_t[y_p == 1]))
    return IoU


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('-t' , '--y_true'   , type=str,  default='y_true', help='y_true image directory')
    parser.add_argument('-p' , '--y_pred'   , type=str,  default='y_pred', help='y_pred image directory')
    parser.add_argument('-th' , '--thresh'   , type=str,  default='0.1', help='y_pred image directory')

    args = parser.parse_args()

    y_trues,image_files = load(args.y_true)
    y_preds,_ = load(args.y_pred)
    dices = []
    IoUs = []
    for i,image_file in enumerate(image_files):
       y_trues[i][y_trues[i] < float(args.thresh)*255] = 0       
       y_trues[i][y_trues[i] >= float(args.thresh)*255] = 1
       y_preds[i][y_preds[i] < float(args.thresh)*255] = 0
       y_preds[i][y_preds[i] >= float(args.thresh)*255] = 1

       dices.append(dice(y_trues[i],y_preds[i]))
       IoUs.append(IoU(y_trues[i],y_preds[i]))
       print(str(IoUs[i]) +","+ str(dices[i]) +"," + image_file)
       print(y_preds[i])
    print("IoU:",statistics.mean(IoUs),"±",statistics.pstdev(IoUs),"Dice:",statistics.mean(dices),"±",statistics.pstdev(dices))
   











