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

# avoid the odd behavior of pickle by importing under a different name
import pcanet as net
from utilstmp import load_model, save_model, load_mnist, set_device


parser = argparse.ArgumentParser(description="PCANet example")
parser.add_argument("--gpu", "-g", type=int, default=1,
                    help="GPU ID (negative value indicates CPU)")

parser.add_argument('--mode',dest="mode",default="train",
                                   help='Choice of train/test mode')
#subparsers.required = True
#train_parser = subparsers.add_parser("train")
#train_parser.add_argument("--out", "-o", default="result",
#                          help="Directory to output the result")

#test_parser = subparsers.add_parser("test")
#test_parser.add_argument("--pretrained-model", default="result",
#                         dest="pretrained_model",
#                         help="Directory containing the trained model")

args = parser.parse_args()


def train(train_loader):
    images_train=np.zeros(shape=(1,1,28,28))
    y_train=np.zeros(shape=(1,))
    for images, labels in train_loader:  # test set 批处理
        images_train= np.concatenate((images_train,images.numpy()),axis=0)
        y_train=np.concatenate((y_train,labels.numpy()),axis=0)
    """ 
    N=10 images_train=np.random.randn(N,28,28,1)
    y_train=np.random.randint(2,size=(N,1))"""
    

    print("Training PCANet")

    pcanet = net.PCANet(
        image_shape=28,
        filter_shape_l1=2, step_shape_l1=1, n_l1_output=3,
        filter_shape_l2=2, step_shape_l2=1, n_l2_output=3,
        filter_shape_pooling=2, step_shape_pooling=2)
         
    pcanet.validate_structure()

    t1 = timeit.default_timer()
    pcanet.fit(images_train)
    t2 = timeit.default_timer()

    train_time = t2 - t1

    t1 = timeit.default_timer()
    X_train = pcanet.transform(images_train)
    t2 = timeit.default_timer()

    transform_time = t2 - t1

    print("Training the classifier")

    classifier = SVC(C=10,gamma='auto')
    classifier.fit(X_train, y_train.ravel())
    return pcanet, classifier


def test(pcanet, classifier, test_set):
    images_test, y_test = test_set

    X_test = pcanet.transform(images_test)
    y_pred = classifier.predict(X_test)
    return y_pred, y_test

DataPath='/git/data'
dataset='MNIST'
batch_size=1

train_loader, test_loader = data_loading(DataPath,dataset,batch_size)

if args.gpu >= 0:
    set_device(args.gpu)


if args.mode == "train":
    print("Training the model...")
    pcanet, classifier = train(train_loader)

        
    save_model(pcanet, "pcanet.pkl")
    save_model(classifier, "classifier.pkl")
    print("Model saved")

elif args.mode == "test":
    pcanet = load_model(join(args.pretrained_model, "pcanet.pkl"))
    classifier = load_model(join(args.pretrained_model, "classifier.pkl"))

    y_test, y_pred = test(pcanet, classifier, test_loader)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: {}".format(accuracy))
