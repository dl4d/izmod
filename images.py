#
# Image wrapper for both classification, segmentation and encoder dataset
#
import os
from PIL import Image
import numpy as np
import datetime
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical

import pickle
import random

import matplotlib.pyplot as plt

from flickrapi import FlickrAPI
import pandas as pd
import sys

import csv
import requests
import time



class iz_image_dataset:

    def __init__(self,parameters=None):

        X         = None
        y         = None
        X_test    = None
        y_test    = None
        synsets   = None
        path      = None
        path_target = None
        resize    = None
        type      = None
        name      = "NoName"
        date_creation = None
        test_split = None

        if parameters is not None:

            if "path" not in parameters:
                print('IZ Import Error: No PATH provided !')
                return

            if "path_target" not in parameters:
                parameters["path_target"] = None

            if "resize" not in parameters:
                parameters["resize"] = (32,32)

            if "name" not in parameters:
                parameters["name"]="NoName"

            if "type" not in parameters:
                parameters["type"]=None

            self.iz_import(
                parameters["path"],
                parameters["path_target"],
                parameters["resize"],
                parameters["name"],
                parameters["type"]
                )



    def infos(self):
        if hasattr(self,"type"):
            if self.type=="classification":
                self.infos_classification()
                return None
            if (self.type=="segmentation") or (self.type=="encoder"):
                self.infos_segmentation_or_encoder()
                return None
        else:
            print('Warning: Empty IZ object, no infos to show !')

    def infos_classification(self):

        print("\n")
        print("|----------------------------------------------------")
        print("| ¤¤¤¤¤¤¤¤¤ IZ Image Dataset informations ¤¤¤¤¤¤¤¤¤  ")
        print("|----------------------------------------------------")
        print("| Name                       | ", self.name)
        print("| Type                       | ", self.type)
        print("| Date Creation              | ", self.date_creation)
        print("| Path                       | ", self.path)
        print("|----------------------------------------------------")
        print("| Number of images           | ", self.X.shape[0])
        print("| Number of distinct classes | ", len(np.unique(self.y)))
        print("| Occurences by classes      | ", self.count_occurences_by_class())
        print("| Synsets:                   | ", self.synsets)
        print("|----------------------------------------------------")
        print("| Image channels             | ", self.X.shape[3])
        print("| Image size                 | ", self.resize)
        print("| Min-Maxcolor intensity     | [",self.X.min(),",",self.X.max(),"]")
        print("| Image BitDepths            | ", self.get_bitdepth(self.X))
        print("|----------------------------------------------------")
        print("|----------------------------------------------------")
        print("| ¤¤¤¤¤¤¤¤¤   IZ Image Dataset Splitting    ¤¤¤¤¤¤¤¤¤  ")
        print("|----------------------------------------------------")
        if hasattr(self,"X_"):
                print("| IZ Split : True")
                print("|----------------------------------------------------")
                print("| [For Training] Number of images   | ", self.X_.shape[0])
                print("| [For Testing ]  Number of images  | ", self.X_test.shape[0])
        else:
                print("| IZ Split : False")

        print("\n")


    def infos_segmentation_or_encoder(self):

        print("\n")
        print("|----------------------------------------------------")
        print("| ¤¤¤¤¤¤¤¤¤ IZ Image Dataset informations ¤¤¤¤¤¤¤¤¤  ")
        print("|----------------------------------------------------")
        print("| Name                           | ", self.name)
        print("| Type                           | ", self.type)
        print("| Date Creation                  | ", self.date_creation)
        print("| Path                           | ", self.path)
        print("| Path Target                    | ", self.path_target)
        print("|----------------------------------------------------")
        print("| Number of images               | ", self.X.shape[0])
        print("| Number of masks                | ", self.y.shape[0])
        print("|----------------------------------------------------")
        print("| Image channels                 | ", self.X.shape[3])
        print("| Target channels                | ", self.y.shape[3])
        print("| Image size                     | ", self.resize)
        print("| Image Min-Maxcolor intensity   | [",self.X.min(),",",self.X.max(),"]")
        print("| Target  Min-Maxcolor intensity | [",self.y.min(),",",self.y.max(),"]")
        print("| Image BitDepths                | ", self.get_bitdepth(self.X))
        print("| Target BitDepths               | ", self.get_bitdepth(self.y))
        print("|----------------------------------------------------")

        print("\n")

    def get_bitdepth(self,data):

        if np.unique(data[0]).shape[0]==2:
            return "binary"
        if data.max()<(2**8):
            return "8"
        if data.max()<(2**16):
            return "16"
        if data.max()<(2**32):
            return "32"
        if data.max()<(2**64):
            return "64"

    def count_occurences_by_class(self):
        if hasattr(self,"y_scalers_type"):
            if self.y_scalers_type == "categorical":
                return self.y_.shape[1]
            else:
                return Counter(self.y)
        else:
            return Counter(self.y)

    def iz_import(self,path=None,path_target=None,resize=None,name=None,type=None):

        if path is None:
            print('IZ Import Error: No PATH provided !')
            return None

        if path_target is None:
            self.iz_import_classification(path,resize,name=name)
        else:
            if type is None:
                self.iz_import_segmentation_or_encoder(path,path_target,resize,name=name)
            else:
                self.iz_import_segmentation_or_encoder(path,path_target,resize,name=name,type="encoder")

        return None

    def iz_import_classification(self,path,resize0,name):

        print('[IZ Import] Import images for classification')
        print('[IZ Import] from   : ', path)
        print('[IZ Import] resize : ', str(resize0))

        images  = []
        labels  = []
        synsets = []
        img_rows=resize0[0]
        img_cols=resize0[1]
        print("[IZ Import] Reading "+path)

        k=0
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                curdir = os.path.join(path,d)
                for filename in os.listdir(curdir):
                    curimg = os.path.join(curdir, filename)
                    img = Image.open(curimg)
                    resize = img.resize((img_rows,img_cols), Image.NEAREST)
                    images.append(resize)
                    labels.append(k)
                synsets.append(d)
                k=k+1
        imgarray=list();
        for i in range(len(images)):
            tmp = np.array(images[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        synsets=dict(enumerate(np.unique(synsets)))


        X = imgarray.astype('float32')

        if len(X.shape)==3:
            X = np.expand_dims(X,axis=3)

        Y = np.asarray(labels)
        synsets = synsets

        self.X = X
        self.y = Y
        self.synsets = synsets
        self.path = path
        self.resize = resize0
        self.type = "classification"
        self.name = name
        self.date_creation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


    def iz_import_segmentation_or_encoder(self,path,path_target,resize0,name,type="segmentation"):

        print('[IZ Import] Import images for ', type)
        print('[IZ Import] from images path   : ', path)
        print('[IZ Import] from target path   : ', path_target)
        print('[IZ Import] resize             : ', str(resize0))

        images  = []
        labels  = []
        img_rows=resize0[0]
        img_cols=resize0[1]
        for filename in sorted(os.listdir(path)):
            curimg = os.path.join(path, filename)
            img = Image.open(curimg)
            resize = img.resize((img_rows,img_cols), Image.NEAREST)
            images.append(resize)
        imgarray=list();
        for i in range(len(images)):
            tmp = np.array(images[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)


        for filename in sorted(os.listdir(path_target)):
            curimg = os.path.join(path_target, filename)
            img = Image.open(curimg)
            resize = img.resize((img_rows,img_cols), Image.NEAREST)
            labels.append(resize)
        labarray=list();
        for i in range(len(labels)):
            tmp = np.array(labels[i])
            labarray.append(tmp)
        labarray = np.asarray(labarray)

        X = imgarray.astype('float32')
        Y = labarray.astype('float32')

        if len(X.shape)==3:
            X = np.expand_dims(X,axis=3)
        if len(Y.shape)==3:
            Y = np.expand_dims(Y,axis=3)


        self.X = X
        self.y = Y
        self.path = path
        self.path_target = path_target
        self.resize = resize0
        self.type = type
        self.name = name
        self.date_creation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


    def save(self,filename):
        filehandler = open(filename,"wb")
        pickle.dump(self,filehandler)


    def show_random_images(self):
        print ("Still TODO ....")
        pass


    def rm_rf(self,d):
        for path in (os.path.join(d,f) for f in os.listdir(d)):
            if os.path.isdir(path):
                self.rm_rf(path)
            else:
                os.unlink(path)
        os.rmdir(d)
    #Test flickr API

    def iz_import_from_flickr(self,parameters,tags,count,delete=False):

        if parameters is not None:
            if "path" not in parameters:
                print('IZ Import Error: No PATH provided !')
                return
            if "key" not in parameters:
                print('"key" not in parameter list ! Please provide your Flickr API Key !')
                return
            if "secret" not in parameters:
                print('"secret" not in parameter list ! Please provide your Flickr API Secret !')

        os.mkdir(parameters["path"])

        for tag in tags:
            urls = self.get_urls(parameters,tag,count)
            self.put_images(parameters,tag+"_urls.csv")

        if parameters is not None:

            if "path" not in parameters:
                print('IZ Import Error: No PATH provided !')
                return

            if "path_target" not in parameters:
                parameters["path_target"] = None

            if "resize" not in parameters:
                parameters["resize"] = (32,32)

            if "name" not in parameters:
                parameters["name"]="NoName"

            if "type" not in parameters:
                parameters["type"]=None

            self.iz_import(
                parameters["path"],
                parameters["path_target"],
                parameters["resize"],
                parameters["name"],
                parameters["type"]
                )
        if delete:
            self.rm_rf(parameters["path"])



    #Source inspired: https://towardsdatascience.com/how-to-use-flickr-api-to-collect-data-for-deep-learning-experiments-209b55a09628
    def get_urls(self,parameters,image_tag,MAX_COUNT):

        key = parameters["key"]
        secret=parameters["secret"]

        flickr = FlickrAPI(key, secret)
        photos = flickr.walk(text=image_tag,
                                tag_mode='all',
                                tags=image_tag,
                                extras='url_o',
                                per_page=50,
                                sort='relevance')
        count=0
        urls=[]
        for photo in photos:
            if count< MAX_COUNT:
                count=count+1
                print("Fetching url for image number {}".format(count))
                try:
                    url=photo.get('url_o')
                    urls.append(url)
                except:
                    print("Url for image number {} could not be fetched".format(count))
            else:
                print("Done fetching urls, fetched {} urls out of {}".format(len(urls),MAX_COUNT))
                break
        urls=pd.Series(urls)
        print("Writing out the urls in the current directory")
        urls.to_csv(os.path.join(parameters["path"],image_tag+"_urls.csv"))
        print("Done!!!")
        return urls

    #Source inspired: https://towardsdatascience.com/how-to-use-flickr-api-to-collect-data-for-deep-learning-experiments-209b55a09628
    def put_images(self,parameters,FILE_NAME):
        urls=[]
        with open(os.path.join(parameters["path"],FILE_NAME),newline="") as csvfile:
            doc=csv.reader(csvfile,delimiter=",")
            for row in doc:
                if row[1].startswith("https"):
                    urls.append(row[1])
        #if not os.path.isdir(os.path.join(os.getcwd(),FILE_NAME.split("_")[0])):
        if not os.path.isdir(os.path.join(parameters["path"],FILE_NAME.split("_")[0])):
            os.mkdir(os.path.join(parameters["path"],FILE_NAME.split("_")[0]))
        t0=time.time()
        for url in enumerate(urls):
            print("Starting download {} of ".format(url[0]+1),len(urls))
            try:
                resp=requests.get(url[1],stream=True)
                #path_to_write=os.path.join(os.getcwd(),FILE_NAME.split("_")[0],url[1].split("/")[-1])
                path_to_write=os.path.join(parameters["path"],FILE_NAME.split("_")[0],url[1].split("/")[-1])
                outfile=open(path_to_write,'wb')
                outfile.write(resp.content)
                outfile.close()
                print("Done downloading {} of {}".format(url[0]+1,len(urls)))
            except:
                print("Failed to download url number {}".format(url[0]))
        t1=time.time()
        print("Done with download, job took {} seconds".format(t1-t0))


    def split(self,test=0.2,random_state=42):
        self.X,self.X_test,self.y,self.y_test = train_test_split(self.X,self.y,test_size=test,random_state=random_state)


    def input_shape(self):
        return self.X.shape[1:]
    #
    # def output_neurons(self):
    #     #return self.y.shape
    #     if self.type == "classification":
    #         if hasattr(self,"y_scalers_type"):
    #             if self.y_scalers_type == "categorical":
    #                 return self.y_.shape[1]
    #             else:
    #                 return 1
    #     if self.type == "segmentation":
    #         return self.input_shape()
    #
    #     if self.type == "encoder":
    #         return self.input_shape()
