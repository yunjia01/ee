# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and setting
#


# import numpy as np
# import matplotlib.pyplot as plt
#
# x=np.arange(0,9,0.1)
# y=np.sin(x)
# y1=np.cos(x)
# plt.plot(x,y,label="sin")
# plt.plot(x,y1,linestyle="--",label="cos")
# plt.title("sin & cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.image import imread
# img=imread("C:\\Users\\11208\\Desktop\\modbus.jpg")
# plt.imshow(img)
# plt.show()


# def AND(x,y):
#     w1,w2,yz=2,2,3
#     temp=x*w1+y*w2
#     if temp>yz:
#         return 1
#     else:
#         return 0
# print(AND(1,1))


# import numpy as np
#
# def AND(x1,x2):
#     x=np.array([x1,x2])
#     w=np.array([0.5,0.5])
#     b=-0.7
#     temp=np.sum(x*w)+b
#     if temp>0:
#         return 1
#     else:
#         return 0
# print(AND(1,1))

# def OR(x3,x4):
#     x=np.array([x3,x4])
#     w=np.array([2,2])
#     b=-1
#     temp=np.sum(x*w)+b
#     if temp>0:
#         return 1
#     else:
#         return 0
# print(OR(0,0))
import pickle

import numpy as np

# def nand(x1,x2):
#     x=np.array([x1,x2])
#     w=np.array([-2,-2])
#     b=1
#     temp=np.sum(x*w)+b
#     if temp>0:
#         return 1
#     else:
#         return 0
# print(nand(0,0))

#
# import numpy as np
#
# def AND(x1,x2):
#     X1=np.array([x1,x2])
#     W1=np.array([2,2])
#     b1=-3
#     temp1=np.sum(X1*W1)+b1
#     if temp1>0:
#         return 1
#     else: return 0
#
# def nand(x3,x4):
#     X2=np.array([x3,x4])
#     W2=np.array([-3,-3])
#     b2=4
#     temp2=np.sum(X2*W2)+b2
#     if temp2>0:
#         return 1
#     else: return 0
#
#
# def OR(x5,x6):
#     X3=np.array([x5,x6])
#     W3=np.array([2,2])
#     b3=-1
#     temp=np.sum(X3*W3)+b3
#     if temp>0:
#         return 1
#     else:
#         return 0
#
# def xorr(s1,s2):
#     bb = OR(s1, s2)
#     aa= nand(s1, s2)
#     y=AND(bb,aa)
#     print(y)
# xorr(1,1)


# import numpy as np
# import matplotlib.pyplot as plt
#
# def funtion(x):
#     return np.array(x>0,dtype=np.int_)
#
# x=np.arange(-5,5,0.1)
# y=funtion(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
# def sigmoid(x):
#     return (1/(np.exp(-x)+1))
#
# x=np.arange(-5,5,0.1)
# y=np.array(sigmoid(x))
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
# def rel(x):
#     return np.maximum(0,x)
#
# x=np.arange(-5,5,0.1)
# y=rel(x)
# plt.plot(x,y,linestyle="--",label="rel")
# plt.legend()
# plt.ylim(-2,2)
# plt.show()


# import numpy as np
#
# x=np.array([[1,2],[3,4]])
# y=np.array([[5,6],[7,6]])
# z=np.dot(x,y)
# print(np.shape(z ))



# import numpy as np
#
# x=np.array([1,2])
# w=np.array([[1,2,3],[4,5,6]])
# z=np.dot(x,w)
# print(z)


#
# def network():
#     network={}
#     network['w1']=np.array([[1,3,2],[2,3,1]])
#     network['b1'] = np.array([[1, 2, 8]])
#     network['w2']=np.array([[1,4],[4,2],[9,8]])
#     network['b2'] = np.array([[1, 3]])
#     network['w3']=np.array([[2,3],[2,1]])
#     network['b3'] = np.array([[2, 3]])
#     return network
#
# def sigmoid(a):
#     y=np.maximum(0,a)
#     return y
#
#
# def funtion(a):
#     return a
#
# def forward(x,network):
#     w1,w2,w3=network['w1'],network['w2'],network['w3']
#     b1,b2,b3=network['b1'],network['b2'],network['b3']

#     a1=np.dot(x,w1)+b1
#     z1=sigmoid(a1)
#     a2=np.dot(z1,w2)+b2
#     z2=sigmoid(a2)
#     a3=np.dot(z2,w3)+b3
#     y=funtion(a3)
#     return y
#
# network=network()
# x=np.array([1,2])
# y=forward(x,network)
#
# print(y)



# a=np.array([1,2,3])
# exp_a=np.exp(a)
# y=exp_a/np.sum(exp_a)
# print(y)

#
# a=np.array([1,2,3])
# def softmax(a):
#     exp_a=np.exp(a)
#     y=exp_a/np.sum(exp_a)
#     return y
# z=softmax(a)
# print(z)



# a=np.array([1,2,3])
# def softmax(a):
#     max_a=np.max(a)
#     a1=np.exp(a-max_a)
#     y=a1/np.sum(a1)
#     return y
# aa=softmax(a)
# print(aa)



# import sys,os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# import numpy as np
# from PIL import Image
#
# def img_show(img):
#     pil_img=Image.fromarray(img)
#     pil_img.show()
# (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
# img=x_train[100]
# label=t_train[100]
# print(label)
#
# print(img.shape)
# img=img.reshape(28,28)
# print(img.shape)
# img_show(img)



import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open(r"D:\python\【源代码】深度学习入门：基于Python的理论与实现\ch03\sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def softmax(a):
    y=np.exp(a-np.max(a))/np.sum(np.exp(a-np.max(a)))
    return y

def sigmoid(a):
    return 1/(1+np.exp(-a))

def predict(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax(a3)
    return y
x,t=get_data()
network = init_network()

accuracy=0
for i in range(len(x)):
    y = predict(network, x[i])
    z = np.argmax(y)
    if z==t[i]:
        accuracy+=1
print("accuracy:"+str(float(accuracy)/len(x)))














