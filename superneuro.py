import random
import numpy as np

INPUT_DIM = 4 #Входные признаки для нейросети
OUT_DIM = 3 #Определение класса ирисов
H_DIM = 5 #Количество нейронов в первом слое

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out/np.sum(out)

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

x = np.random.randn(1, INPUT_DIM)#Признаки ириса
y = random.randint(0, OUT_DIM)#Правильный ответ

W1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(1, H_DIM)

W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(1, OUT_DIM)

#Forward
t1 = x @ W1 + b1
h1 = relu(t1)
t2 = h1 @ W2 + b2
z = softmax(t2)
E = sparse_cross_entropy(z, y)

#Backward
y_full = to_full(y, OUT_DIM)

