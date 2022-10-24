import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from ExampleClassifierModule import ExampleClassifier

import numpy as np


# 채널 별 mean 계산
def get_mean(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1, 2)) for image, _ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]


# 채널 별 str 계산
def get_std(dataset):
    stdRGB = [np.std(image.numpy(), axis=(1, 2)) for image, _ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]


class DLCClassifier(ExampleClassifier):
    def __init__(self, path_data, pretrain=None):  # path data = dataset/train/ , pretrain = None
        super(DLCClassifier,self).__init__(path_data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 학습 환경 설정

        # Please refer to dataset directory structure
        self.path_data = path_data

        if pretrain is not None:
            # For evaluation
            self.model = torch.load(pretrain)
            self.model = self.model.to(self.device)
        else:
            self.build_model()

        # Dataset loading에 적용하기 위한 transform은 생성자에서 선언

        # 데이터셋 구조가 정해져 있으므로, ImageFolder class를 사용하기를 추천
        # 다른 class를 사용 할 경우 반드시 각 샘플은 (image, label)을 반환해야 함.
        self.dataset = ImageFolder(
            self.path_data,
            transform=transforms.ToTensor()

        )
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(get_mean(self.dataset), get_std(self.dataset))
        ])

        self.dataset.transform = self.transforms

        self.num_data = len(self.dataset)

    def build_model(self):
        self.model = models.resnet50(pretrained=True)  # true 옵션으로 사전 학습된 모델을 로드

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 5)
        self.model = self.model.to(self.device)

    def forward(self, x):
        y = self.model(x)
        return y

    def train_model(self, config):
        batch_size = config['batch_size']
        epochs = config['epochs']
        loss = config['loss']
        optim = config['optim']

        train_loader = DataLoader(self.dataset, batch_size=batch_size,shuffle=True)

        print('Number of data : {}'.format(self.num_data))

        # 가장 기본적인 훈련 코드 구현의 예시
        self.model.train()
        for epoch in range(1, epochs+1):
            print('Epoch : {} / {}'.format(epoch, epochs))

            # 진행 상황을 보기 위한 tqdm 이용 예시
            pbar = tqdm(total=self.num_data, dynamic_ncols=True)

            for batch, sample in enumerate(train_loader):
                img, label = sample
                img, label = img.to(self.device), label.to(self.device)
                optim.zero_grad()

                output = self.forward(img)

                loss_val = loss(output, label)

                loss_val.backward()

                optim.step()

                pbar.set_description('Loss : {}'.format(loss_val.item()))
                pbar.update(batch_size)

            pbar.close()


    def eval_model(self):
        super().eval_model()
