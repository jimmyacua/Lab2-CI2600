import csv
import torch
import pandas as pd
from matplotlib import pyplot as plt
import torch.optim as optim
import math


def read_csv(filename, col1, col2):
    my_csv = pd.read_csv(filename)
    column1 = my_csv[col1]
    column2 = my_csv[col2]
    tensor1 = torch.Tensor(list(column1))
    tensor2 = torch.Tensor(list(column2))
    return tensor1, tensor2


def model(x, w, b):
    y = w * x + b
    return y


def loss_fn(y, oov):
    squared_diff = (oov-y)**2
    return squared_diff.mean()


def dmodel_w(y, w, b):
    return y


def dmodel_b(y, w, b):
    return 1.0


def dloss_m(y, oov):
    d_dif = 2 * (y-oov)
    return d_dif


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_m(t_p, t_c) * dmodel_w(t_u, w, b)
    dloss_db = dloss_m(t_p, t_c) * dmodel_b(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])


def training(n, w, b,  alpha, x, y):
    for i in range(0, n):
        '''if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()'''

        t_p = model(x, w, b)
        loss = loss_fn(t_p, y)
        optimizer = optim.SGD([w, b], lr=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ya no se usa por el auto grad
        '''grad = grad_fn(x, y, t_p, w, b)
        w = w - alpha*grad[0]
        b = b - alpha * grad[1]'''

        #w = (w - learning_rate * w.grad).detach().requires_grad_()
        #b = (b - learning_rate * b.grad).detach().requires_grad_()

        if n % 500 == 0:
            print('Epoch %d, Loss %f' % (i, float(loss)))
        '''print('Epoch %d, Loss %f' % (i, float(loss)))
        print('w: ', w, ', b:', b)
        print('Grad: ', grad)'''

    return w, b


def model_nonlin(a, x, b, c):
    y = a*x**b+c  # para usar esta formula usar le = 1e-6
    # y = a*(math.e**(b*x))+c  # es mejor la primera
    return y


def training_nonlin(n, w, b, c, alpha, x, y):
    for i in range(0, n):
        t_p = model_nonlin(w, x, b, c)
        loss = loss_fn(t_p, y)
        optimizer = optim.SGD([w, b], lr=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 500 == 0:
            print('Epoch %d, Loss %f' % (i, float(loss)))


if __name__ == "__main__":
    tensor1, tensor2 = read_csv('SkillCraft1_Dataset.csv', 'ActionLatency', 'APM')

    '''Tensor 1 '''
    testSet1_tensor1 = tensor1[0:30]
    rest1 = torch.cat([tensor1[30:]])
    indixes_tensor1 = torch.randperm(rest1.size()[0])
    twentyPercent1 = int(rest1.size()[0]*0.2)

    aux = indixes_tensor1[0:twentyPercent1]
    testSet_tensor1 = rest1[aux]

    aux = indixes_tensor1[twentyPercent1:]
    trainingSet_tensor1 = rest1[aux]

    '''Tensor 2 '''
    testSet1_tensor2 = tensor2[0:30]
    rest2 = torch.cat([tensor2[30:]])
    indixes_tensor2 = torch.randperm(rest2.size()[0])
    twentyPercent2 = int(rest2.size()[0] * 0.2)
    aux = indixes_tensor2[0:twentyPercent2]
    testSet_tensor2 = rest2[aux]

    aux = indixes_tensor2[twentyPercent2:]
    trainingSet_tensor2 = rest2[aux]

    w = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    learning_rate = 1e-9

    xn = (trainingSet_tensor1 - trainingSet_tensor1.mean())/trainingSet_tensor1.size()[0]# para normalizar la entrada.
    #xn = 0.1 * trainingSet_tensor1
    t_p = model(xn, w, b)
            #t_u = trainingSet_tensor1, tc = trainingSet_tensor2
    plt.plot(trainingSet_tensor1.numpy(), t_p.detach().numpy())
    plt.plot(trainingSet_tensor1.numpy(), trainingSet_tensor2.numpy(), 'o')
    #training(5000, w, b, learning_rate, xn, trainingSet_tensor2)
    training_nonlin(2000, w, b, c, learning_rate, trainingSet_tensor1, trainingSet_tensor2)