import os,sys
from os.path import isdir, join
import timeit
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
DataPath='/git/data'
sys.path.append(DataPath)
from CifarDataLoader import data_loading
import torch
# avoid the odd behavior of pickle by importing under a different name
import pcanet as net
from utils import load_model, save_model, set_device


parser = argparse.ArgumentParser(description="PCANet example")
parser.add_argument("--gpu", "-g", type=int, default=4,
                    help="GPU ID (negative value indicates CPU)")

parser.add_argument('--mode',dest="mode",default="train",
                                   help='Choice of train/test mode')

parser.add_argument('--dataset',dest="dataset",default="MNIST")
parser.add_argument('--batchSize',dest="batchSize",type=int,default=1)
parser.add_argument('--Numlayers',dest="Numlayers",type=int,default=1)


"""subparsers.required = True
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--out", "-o", default="result",
                         help="Directory to output the result")

test_parser = subparsers.add_parser("test")
test_parser.add_argument("--pretrained-model", default="result",
                         dest="pretrained_model",
                         help="Directory containing the trained model")
"""
args = parser.parse_args()

def train(train_loader,dataset,Numlayers):
    """i=0
    for images, labels in train_loader:  # test set 批处理
        if i==0:
            images_train= np.swapaxes(images.numpy(),3,1)
            y_train=labels.numpy()
            i+=1
        else:  
            images_train= np.concatenate((images_train,np.swapaxes(images.numpy(),3,1)),axis=0)
            y_train=np.concatenate((y_train,labels.numpy()),axis=0)
    """
    N=10 
    if dataset=='MNIST':
        imageShape=28
    elif dataset=='CIFAR10':
        imageShape=32
    images_train=np.random.randn(N,imageShape,imageShape,1)
    y_train=np.random.randint(2,size=(N,1))
   
    print("Training PCANet")
    
    if dataset=='MNIST':
        pcanet = net.PCANet(
        image_shape=28,
        filter_shape_l1=2, step_shape_l1=1, n_l1_output=3,
        filter_shape_l2=2, step_shape_l2=1, n_l2_output=3,
        filter_shape_pooling=2, step_shape_pooling=2)
        
    elif dataset=='CIFAR10':
        pcanet = net.PCANet(
        image_shape= 32,
        filter_shape_l1=5, step_shape_l1=1, n_l1_output= 16,
        filter_shape_l2=5, step_shape_l2=1, n_l2_output=8,
        filter_shape_pooling=8,  step_shape_pooling=4)
        
    pcanet.validate_structure()
    print("Training the classifier")
    
    for i in range(Numlayers):
        pcanet.fit(images_train)
        X_train = pcanet.transform(images_train)
        images_train = SVC(C=10,gamma='auto')
        classifier.fit(X_train, y_train.ravel())
    return pcanet, classifier


def test(pcanet, classifier, test_loader):
    i=0
    for images, labels in test_loader:  # test set 批处理
        if i==0:
            images_test= np.swapaxes(images.numpy(),3,1)
            y_test=labels.numpy()
            i+=1
        else:
            images_test= np.concatenate((images_test,np.swapaxes(images.numpy(),3,1)),axis=0)
            y_test=np.concatenate((y_test,labels.numpy()),axis=0)
    #images_test, y_test = test_set

    X_test = pcanet.transform(images_test)
    y_pred = classifier.predict(X_test)
    return y_pred, y_test

if __name__ == "__main__":
    DataPath='/git/data'

    train_loader, test_loader = data_loading(DataPath,args.dataset,args.batchSize)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    if args.mode == "train":
        print("Training the model...")
        pcanet, classifier = train(train_loader,args.dataset,args.Numlayers)
        save_model(pcanet, args.dataset+"_pcanet.pkl")
        save_model(classifier, args.dataset+"_classifier.pkl")
        print("Model saved")

    elif args.mode == "test":
        pcanet = load_model(args.dataset+"_pcanet.pkl")
        classifier = load_model(args.dataset+"_classifier.pkl")

        y_test, y_pred = test(pcanet, classifier, test_loader)

        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy: {}".format(accuracy))
