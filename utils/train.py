import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import numpy as np
import os
from utils.data import load_data, getMap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def SingleTrain(net, train_iter, loss, optim, num_classes):
    """train for one epoch

    Args:
        net (nn.module): training model
        train_iter (dataloader): iterator of training set
        loss (): loss function of the model
        optim (): optimizer of the model

    Returns:
        float: loss value
    """
    net.train() # 开启训练模式
    # 将计算累计loss值的变量定义在GPU上，无需在计算时在CPU与GPU之间移动，耗费时间
    los = torch.zeros(1).cuda()
    correct = torch.zeros(1).cuda()
    total = 0
    for _,(x, y) in tqdm(enumerate(train_iter), desc="Training", total=len(train_iter)):
        # 将数据迁移到GPU
        x, y = x.to(device), y.to(device)
        # 清零梯度
        optim.zero_grad()
        # 向前传播，输出预测标签
        haty = net(x)
        haty = haty.reshape(-1, num_classes)
        # 计算损失值
        l = loss(haty,y)
        # 反向传播，计算得到每个参数的梯度值
        l.backward()
        # 梯度下降，由优化器更新参数
        optim.step()
        # 累计损失值
        los += l
        correct += torch.eq(torch.max(haty.data, 1)[1], y).sum()
        total += y.size(0)
    los = los.item() / total
    acc = correct.item() / total
    
    return los, acc


    
@torch.no_grad() # 使新增的tensor没有梯度，使带梯度的tensor能够进行原地运算
def score(net, data_iter, loss, num_classes):
    net.eval() # 开启评估模式
    # 将计算累计正确预测样例数的变量定义在GPU上，无需在计算时在CPU与GPU之间移动，耗费时间
    total = 0
    correct = torch.zeros(1).cuda()
    los = torch.zeros(1).cuda()
    test = 0
    test2 = 0
    for _, (x,y) in tqdm(enumerate(data_iter), desc="Validating", total=len(data_iter)):
        x, y = x.to(device), y.to(device)
        haty = net(x)
        haty = haty.reshape(-1, num_classes)
        los += loss(haty, y)
        # 计算预测标签一致的样例数
        correct += torch.eq(torch.max(haty.data, 1)[1], y).sum()
        total += y.size(0)
    los = los.item() / len(data_iter)
    acc = correct.item() / total
    print(f"{test}/{total}", f"{test2}/{total}")
    return los, acc

@torch.no_grad() # 使新增的tensor没有梯度，使带梯度的tensor能够进行原地运算
def predict(model, net, pt, dataset, img):
    import torchvision.transforms as transforms
    from PIL import Image
    net.eval() # 开启评估模式，非常重要！
    print(f"Loading {model} parameters...")
    net = net.to(device)
    try:
        # 加载权重
        state_dict = torch.load(pt)
        # 导入权重到模型
        net.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR(Fail to load parameters):{e}")
        return False, "ERROR: Fail to load parameters"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 需要保持数据预处理的同等处理条件！
    ])
    x = transform(Image.open(img)).unsqueeze(0)
    x = x.to(device)
    # 向前传播，输出预测标签
    haty = net(x)
    haty = torch.softmax(haty, dim=1)
    # 获取概率最高的类别索引
    index = int(torch.argmax(haty, dim=1))
    label = getMap(dataset).inv[index]
    print(img, index, label)
    return label

def train(model, net, batch, data, learning_rate, epoch, num_classes, pretrained, log):
    if not os.path.exists(log):
        os.mkdir(log)
    test_iter, train_iter, valid_iter = load_data(batch, data)
    valid_min_loss = np.Inf
    count = 0
    net = net.to(device)
    #损失函数
    loss = nn.CrossEntropyLoss()
    #优化函数
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate, momentum=0.9, weight_decay=5e-3) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 35, 60], gamma=0.6)
    start = time()
    dir = f"checkpoint{start}_{model}"
    pt = f'models/{model}_CRC.pt'
    if pretrained:
        print("Loading pretrain parameters...")
        try:
            # 加载权重
            state_dict = torch.load(pt)
            # 导入权重到模型
            net.load_state_dict(state_dict)
        except Exception as e:
            print(f"ERROR(Fail to load parameters):{e}")
    pt = f'{log}/{dir}/{model}_CRC.pt'
    train_loss = []
    test_loss = []
    valid_loss = []
    valid = []
    train = []
    test = []
    print('Start Training...')
    # flag = False
    for i in range(epoch):
        if count == 5:
            count = 0
            optimizer.param_groups[0]["lr"] *= 0.5
            if optimizer.param_groups[0]["lr"] < 1e-5:
                print("Eearly Stop!")
                break
            # 加载权重
            state_dict = torch.load(pt)
            # 导入权重到模型
            net.load_state_dict(state_dict)
            
        print(f'Epoch: {i+1} / {epoch}')
        p = {'lr':round(optimizer.param_groups[0]["lr"],5),
                'weight_decay':round(optimizer.param_groups[0]["weight_decay"], 5),
                'count':count}
        print(f'Current Optimizer Parameters: {p}')
        t = time()
        l, train_acc = SingleTrain(net, train_iter, loss, optimizer, num_classes)
        test_l, test_acc = score(net, test_iter, loss, num_classes)
        valid_l, valid_acc = score(net, valid_iter, loss, num_classes)
        scheduler.step()
        train_loss.append(l)
        train.append(train_acc)
        
        valid.append(valid_acc)
        valid_loss.append(valid_l)
        if valid_min_loss < valid_l:
            count += 1
        else:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_min_loss, valid_l))
            if not os.path.exists(f'{log}/{dir}'):
                os.mkdir(f'{log}/{dir}')
            torch.save(net.state_dict(), pt)
            valid_min_loss = valid_l
            count = 0
            
        test.append(test_acc)
        test_loss.append(test_l)

        print(f'Train Loss:{round(train_loss[-1],3)} Accuracy: {round(train[-1]*100, 3)}%')
        print(f'Valid Loss:{round(valid_loss[-1],3)} Accuracy: {round(valid[-1]*100, 3)}%')
        print(f'Test Loss:{round(test_loss[-1],3)} Accuracy: {round(test[-1]*100, 3)}%')
        print(f'Time for this epoch: {round(time()-t,2)}s\n')
    
    print(f'TRAINING FINISHED!!!!!!!!!!!')
    para = {'architecture': model,
            'optimizer': 'SGD',
            'batch size': batch,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'pretrained': pretrained}
    print(f'Training Parameters: {para}')
    print(f'Time: {round(time()-start,2)}s for {epoch} epoches.\nAverage {round((time()-start)/epoch,2)}s for each.')
    print(f'The optimal train accuracy: {round(max(train)*100,3)}% at epoch {train.index(max(train))}')
    print(f'The optimal test accuracy: {round(max(test)*100,3)}% at epoch {test.index(max(test))}')
    

    
    stat = {
        'para': para,
        'min_train_loss': f'{round(min(train_loss),3)} at epoch {train_loss.index(min(train_loss))}',
        'max_train_acc': f'{round(max(train)*100,3)} at epoch {train.index(max(train))}',
        'min_valid_loss': f'{round(min(valid_loss),3)} at epoch {valid_loss.index(min(valid_loss))}',
        'max_valid_acc': f'{round(max(valid)*100,3)} at epoch {valid.index(max(valid))}',
        'min_test_loss': f'{round(min(test_loss),3)} at epoch {test_loss.index(min(test_loss))}',
        'max_test_acc': f'{round(max(test)*100,3)} at epoch {test.index(max(test))}',
        'time': f"{round((time()-start)/epoch,2)}s / epoch"
    }
    
    
    with open(f'{log}/{dir}/log.txt', 'w') as f:
        for key, value in stat.items():
            f.write(f"{key}: {value}\n")
        
    result = {
        'train': (train_loss, train),
        'test': (test_loss, test),
        'valid': (valid_loss, valid)
    }
    
    for key, value in result.items():
        np.save(f'{log}/{dir}/{key}_loss.npy', value[0])
        np.save(f'{log}/{dir}/{key}_acc.npy', value[1])
    
    plt.plot(range(len(train_loss)), train_loss, color='blue', label='Train Loss')
    plt.plot(range(len(valid_loss)), valid_loss, color='green', label='Valid Loss')
    plt.plot(range(len(test_loss)), test_loss, color='purple', label='Test Loss')
    plt.legend(['training set', 'validation set', 'test set'], loc='center right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{model} Loss")
    plt.savefig(f'{log}/{dir}/Loss_{model}.png', dpi=300, bbox_inches='tight')
    
    
    
    plt.figure()
    plt.plot(range(len(train)),train, color='blue', label='Train Accuracy')
    plt.plot(range(len(valid)), valid, color='green', label='Valid Accuracy')
    plt.plot(range(len(test)),test, color='purple', label='Test Accuracy')
    plt.legend(['training set', 'validation set', 'test set'], loc='center right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model} Accuracy")
    plt.ylim((0.4,1))
    plt.savefig(f'{log}/{dir}/Accuracy_{model}.png', dpi=300, bbox_inches='tight')
    
    
    
    plt.show()
