import csv
import torch
import pandas as pd


def read_csv(filename, col1, col2):
    my_csv = pd.read_csv(filename)
    column1 = my_csv[col1]
    column2 = my_csv[col2]
    tensor1 = torch.Tensor(list(column1))
    tensor2 = torch.Tensor(list(column2))
    return tensor1, tensor2


def model(w, x, b):
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


def training(n, w, b, alpha, x, y):
    for i in range(0, n):
        t_p = model(w, x, b)
        loss = loss_fn(t_p, y)
        grad = grad_fn(x, y, t_p, w, b)

        w = w - alpha*grad[0]
        b = b - alpha * grad[1]

        print('Epoca %d, Loss %f' % (i, float(loss)))

    return w, b


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

    w = torch.ones(1)
    b = torch.zeros(1)
    learning_rate = 1e-4

    #print(trainingSet_tensor1.size())
    #print(w.size())

    #print(model(w, trainingSet_tensor1, b))
    training(1000, w, b, learning_rate, trainingSet_tensor1, trainingSet_tensor2)

