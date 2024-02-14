import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as transforms
import os
import random
from zipfile import ZipFile
from bidict import bidict
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def getMap(dataset):
    classes = [f for f in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, f))]
    class_map = {c:i for i,c in enumerate(classes)}
    return bidict(class_map)


def process(dataset, data_path='data/'):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(f"{data_path}/{dataset[:-4]}"):
        os.mkdir(f"{data_path}/{dataset[:-4]}")
    classes = getMap(dataset[:-4])
    file_list = [img for img in ZipFile(dataset, 'r').namelist() if img[-3:] == "tif"]
    random.shuffle(file_list)
    file_list = file_list[:5000]
    train_list = file_list[:int(len(file_list)*0.8)]
    test_list = file_list[int(len(file_list)*0.8):int(len(file_list)*0.9)]
    valid_list = file_list[int(len(file_list)*0.9):]
    transform = transforms.Compose([
        # transforms.RandomCrop(224, padding=4, pad_if_needed=True),
        # 创建旋转变换
        # transforms.RandomRotation((-30, 30)),
        # 随机灰度变换
        transforms.RandomGrayscale(p=0.5),
        # 随机扭曲
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # 随机抖动
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 随机水平翻转
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    for i, imgs in enumerate((test_list, train_list, valid_list)):
        data = []
        label = []
        for j, f in enumerate(imgs):
            # 打开.tif文件
            img = Image.open(f)
            img = transform(img).unsqueeze(0)
            data.append(img)
            # if data is None:
            #     data = img
            # else:
            #     data = torch.cat((data, img), dim=0)
            label.append(classes[f.split('/')[1]])
            # print(j+1, '/', len(imgs), data.shape, label[-1])
            print(j+1, '/', len(imgs), len(data), img.shape, label[-1])
        
        data = torch.cat(data, dim=0).numpy()
        # np.save(f'{data_path}/{dataset[:-4]}/data_{i}.npy',data)
        torch.save(data, f'{data_path}/{dataset[:-4]}/data_{i}.pt')
        label = torch.Tensor(label)
        # label = np.array(label)
        # np.save(f'{data_path}/{dataset[:-4]}/label_{i}.npy', label)
        torch.save(label, f'{data_path}/{dataset[:-4]}/label_{i}.pt')

# process('NCT-CRC-HE-100K.zip')


class CRC(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
def load_data(batch, data_path, mini_sample=False): # 读取数据，返回迭代器
    if mini_sample:
        pass
    else:
        loaders = []
        for i in range(3):
            # data = np.load(f'{data_path}/data_{i}.npy')
            # data = torch.from_numpy(data)
            # label = np.load(f'{data_path}/label_{i}.npy')
            # label = torch.from_numpy(label).to(torch.long)
            seed = 42
            torch.manual_seed(seed)
            data = torch.load(f'{data_path}/data_{i}.pt')
            label = torch.load(f'{data_path}/label_{i}.pt').to(torch.long)
            set = CRC(data, label)
            loader = DataLoader(set, batch_size=batch, sampler=RandomSampler(set, replacement=True, num_samples=len(set), generator=torch.Generator().manual_seed(seed)))
            loaders.append(loader)
    return loaders    

if __name__ == '__main__':
    process('CRC-VAL-HE-7K.zip')