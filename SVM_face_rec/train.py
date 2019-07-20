import cv2
import numpy as np
from tools import load_img,get_HoG
from sklearn import metrics

def trans_validation(dir_name):
    #validation
    img_list=[]
    global HoG_list
    global labels
    #load positive validation samples
    dir_name=dir_name+'/validation'
    #dir_name='train_data/validation'
    load_img(dir_name+'/p',img_list)
    for i in range(len(img_list)):
        labels.append(1)
    #load negtive validation samples
    tmp=len(img_list)
    load_img(dir_name+'/n',img_list)
    for i in range(len(img_list)-tmp):
        labels.append(-1)
    #get HoG features
    HoG_list=[]
    get_HoG(img_list, HoG_list)

def validate(svm,dir_name):
    #SVM
    #svm=cv2.ml.SVM_load('first_train.xml')
    global HoG_list
    global labels
    _,pred=svm.predict(np.array(HoG_list))
    pred=[int(i) for i in pred]
    cur_acc=metrics.accuracy_score(labels,pred)
    print("on validation set,the current accuracy is ",cur_acc)
    return pred,cur_acc

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
    print('extracting HoG feature')
    HoG_list=[]
    get_HoG(img_list, HoG_list)
    #info print
    print('received ',tmp,' positive sample(s)')
    print('received',len(img_list)-tmp,' negtive sample(s)')
    print('start training')
    #train SVM 考虑基于Hard Example对分类器二次训练https://www.xuebuyuan.com/2083806.html
    best_c=0
    best_acc=0
    C=0.005
    while(C<0.3):#0.3
        svm=cv2.ml.SVM_create()
        svm.setC(C)
        svm.setType(cv2.ml.SVM_C_SVC)
        #svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setKernel(cv2.ml.SVM_LINEAR)#cv2.ml.SVM_LINEAR
        svm.train(np.array(HoG_list),cv2.ml.ROW_SAMPLE,np.array(labels))
        _,cur_acc=validate(svm,o_dir_name)
        if(cur_acc>best_acc):
            best_c=C
            best_acc=cur_acc
        C+=0.005
    svm=cv2.ml.SVM_create()
    svm.setC(best_c)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(np.array(HoG_list),cv2.ml.ROW_SAMPLE,np.array(labels))
    svm.save('first_train.xml')
    print('svm data has been saved')


HoG_list=[]
labels=[]
trans_validation('train_data')
train("train_data")
