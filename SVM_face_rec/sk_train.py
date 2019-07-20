from sklearn.svm import LinearSVC
import numpy as np
from tools import load_img,get_HoG
from sklearn import metrics

def validate(svm,dir_name):
    img_list=[]
    HoG_list=[]
    labels=[]
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
    #SVM
    #svm=cv2.ml.SVM_load('first_train.xml')
    pred=svm.predict(np.array(HoG_list))
    cur_acc=metrics.accuracy_score(labels,pred)
    print("on validation set,the current accuracy is ",cur_acc)



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
    svm=LinearSVC()
    svm.fit(np.array(HoG_list),np.array(labels))
    validate(svm,o_dir_name)
    return 

train("train_data")