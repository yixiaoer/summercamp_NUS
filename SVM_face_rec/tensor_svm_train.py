import numpy as np
from tools import load_img,get_HoG
import tensorflow as tf
from sklearn import metrics

def train_svm(train_data,labels):
    a = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output=tf.subtract(tf.matmul(x_data,a),b)
    

def train(o_dir_name):
    dir_name=o_dir_name+'/train'
    labels=[]
    img_list=[]
    #get positive img
    load_img(dir_name+'/p',img_list)
    for i in range(len(img_list)):
        labels.append(1)
    #get negtive img
    tmp=len(img_list)
    load_img(dir_name+'/n',img_list)
    for i in range(len(img_list)-tmp):
        labels.append(-1)
    #get HoG feature list
    HoG_list=[]
    get_HoG(img_list, HoG_list)
    #info print
    print('received ',tmp,' positive sample(s)')
    print('received',len(img_list)-tmp,' negtive sample(s)')
    print('start training')
