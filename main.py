#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import argparse
import numpy as np
import glob
import random
import tensorflow as tf
import keras.backend as K
from keras.backend import tensorflow_backend as backend
from nn.train import train_unet, train_ae, train_ae_unet
from nn.pred import pred_Unet, pred_Proposed
sys.path.append("yolo")
from yolo import YOLO
import yolo_image
import csv 

# config
IMAGE_SIZE = 256
TARGET_CLASS = 'ventricular_septum' # TARGET YOLO CLASS
SEQUENCE     = ["+1","+2","+3","-1","-2","-3"] # time-sequence label
NETWORKS     = {0:"unet only", 1:"VGGback+autoencoder+unet", 2:"autoencoder pretrain & VGGback+autoencoder+unet"}

# seed
os.environ['PYTHONHASHSEED']='0'
np.random.seed(7)
random.seed(7)
tf.set_random_seed(7)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(session)


def train(inpdir, tardir, vggdir, val_inpdir, val_tardir, val_vggdir, modelpath, nw, batch, total_epoch, save_epoch=5, num=3, size=IMAGE_SIZE,VGG_flag=True):

    if nw in NETWORKS:
        print(NETWORKS[nw])
        os.makedirs(os.path.dirname(modelpath),exist_ok=True)
        if   nw == 0:
            train_unet(inpdir, tardir, val_inpdir, val_tardir, modelpath, batch, total_epoch, save_epoch, size)
        elif nw == 1:
            train_ae_unet(inpdir, tardir, vggdir, val_inpdir, val_tardir, val_vggdir, modelpath, batch, total_epoch, save_epoch, num, size, fine_tuning=False, pretrained_modelpath=None,VGG_flag=VGG_flag)
        elif nw == 2:
            premodelpath = modelpath.rsplit(".",1)[0]+"_ae.hdf5"
            train_ae(inpdir, val_inpdir, premodelpath, batch, total_epoch, save_epoch, num, size=size) # save final model as "model_ae.hdf5"
            train_ae_unet(inpdir, tardir, vggdir, val_inpdir, val_tardir, val_vggdir,modelpath, batch, total_epoch, save_epoch, num, size, fine_tuning=True, pretrained_modelpath=premodelpath, VGG_flag=VGG_flag)
    else:
        print(NETWORKS)


def predict(inpdir, outdir, modelpath, nw, batch, num=3, vggdir=None, size=IMAGE_SIZE,VGG_flag=True):
    if nw in NETWORKS:
        os.makedirs(outdir, exist_ok=True)
        print(NETWORKS[nw])
        if   nw == 0:
            pred_Unet(inpdir, outdir, modelpath, batch, size)
        else:
            pred_Proposed(inpdir, vggdir, outdir, modelpath, batch, num, size, VGG_flag=VGG_flag) 
    else:
        print(NETWORKS)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('-in' , '--inpdir'   , type=str,  default='data1', help='Input image directory')
    parser.add_argument('-ex' , '--expdir'   , type=str,  default='./exp-test', help='Trimmed image directory')
    parser.add_argument('-out', '--outdir'   , type=str,  default='./out-test', help='Trimmed image directory')
    parser.add_argument('-mdl', '--model'    , type=str,  default='./model-data/unet_weights.hdf5', help='Model file path')
    parser.add_argument('-nw' , '--network'  , type=int,  default=0,  help='change unet/ae+unet/pretrain ae+unet')
    parser.add_argument('-bs' , '--batchsize', type=int,  default=12, help='training batch size')
    parser.add_argument('-ep' , '--epoch'    , type=int,  default=25, help='training epoch')
    parser.add_argument('-yl' , '--yolo_on'  , action='store_true' , default=False)
    parser.add_argument('-pr' , '--pred_only', action='store_true', default=False)
    parser.add_argument('-VGG' , '--VGG_flag', action='store_true', default=True)
    args = parser.parse_args() 

    if args.network not in NETWORKS:
        print("error : incorrect args.network")
        print(NETWORKS)
        exit(1)

    if args.pred_only:
        splits = ["test"]
    else:
        splits = ["test", "train"]

    with session.as_default():
        for split in splits:
            # join path & make directory
            inpimg = os.path.join(args.inpdir, split+"/image")
            expimg = os.path.join(args.expdir, split+"/image")
            expvgg = None
            if args.network!=0:
                expvgg=expimg+"_vgg"
                os.makedirs(expvgg, exist_ok=True) 
            
            inplab = os.path.join(args.inpdir, split+"/label")
            explab = os.path.join(args.expdir, split+"/label")
        
            # YOLO detection & training
            if args.yolo_on:
                if split == "test":
                    f=open('test-image_yolo.csv',"w")
                    writer = csv.writer(f, quotechar="'")
                # YOLO
                yolo = YOLO(model_path   = "yolo/model_data/trained_weights_final.h5",
                            classes_path = "yolo/model_data/vs_classes.txt",
                            anchors_path = "yolo/model_data/yolo_anchors.txt")

                images = glob.glob(os.path.join(inpimg,"*.png"))
                print(images)
                images = [s for s in images if "_" not in s]
                images.sort()
                print(images)
                # error : no file
                if images==[]:
                    print("No such file or directory: '"+inpimg+"/*.png'")

                for img in images:
                    # get file name
                    name, ext = os.path.splitext(os.path.basename(img))
                    print(img)
                    # YOLO detection
                    box, label, score = yolo.detect_image_box(img)
                    box = box[np.where(label[:,]==TARGET_CLASS)] # extract TARGET_CLASS
                    if box.shape[0]<1:
                        # if TARGET_CLASS is not deteced, go next image.
                        continue
                    elif box.shape[0]>1:
                        # if number of detections is more than 2, select 1 detection that have highest score.
                        score = score[np.where(label[:,]==TARGET_CLASS)] # extract TARGET_CLASS
                        box = box[[np.argmax(score)]] # select high-scored box

                    # trim time-sequential data
                    if args.network!=0:
                        seqimg = glob.glob(os.path.join(inpimg, name+"_*"+ext))
                        seqnum = [s.replace(os.path.join(inpimg, name+"_"), '') for s in seqimg] # indir/name_+1.ext -> +1.ext
                        seqnum = [s.replace(ext,                            '') for s in seqnum] # +1.ext -> +1
                        seqnum.sort()
                        # checkt time sequential data is enough
                        if seqnum == SEQUENCE:
                            for i in range(len(SEQUENCE)):
                                yolo_image.trim(seqimg[i],box,outdir=expimg+"_"+SEQUENCE[i])
                        else:
                            continue
                        # copy input data for VGGbackbone
                        shutil.copy(img,os.path.join(expvgg, name+ext)) 
                    # trim input data
                    yolo_image.trim(img,box,outdir=expimg)
                    # trim annotation data
                    lab = os.path.join(inplab, name+ext)
                    yolo_image.trim(lab,box,outdir=explab)
                    if split == "test":
                        writer.writerow([lab,str(int(box[0][0]/1.2)),str(int(box[0][1]*1.2)),str(int(box[0][2]/1.2)),str(int(box[0][3]*1.2))])
                    

                    


        # train
        if args.pred_only==False:
            # set test folder
            val_expimg = os.path.join(args.expdir, "test/image")
            val_explab = os.path.join(args.expdir, "test/label")
            if args.network!=0:
                val_expvgg=val_expimg+"_vgg"
            else:
                val_expvgg=expvgg

            train(inpdir=expimg, tardir=explab, vggdir=expvgg, val_inpdir=val_expimg, val_tardir=val_explab, val_vggdir=val_expvgg, modelpath=args.model,
                nw=args.network, batch=args.batchsize, total_epoch=args.epoch, VGG_flag=args.VGG_flag)

        predict(inpdir=expimg, vggdir=expvgg, outdir=args.outdir, modelpath=args.model, nw=args.network, batch=args.batchsize, VGG_flag=args.VGG_flag)

    backend.clear_session()
  
