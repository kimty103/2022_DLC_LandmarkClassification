from ExampleClassifierModule import ExampleClassifier


class DLCClassifierModule(ExampleClassifier):
    def __init__(self, path_data, pretrain=None):   # path data = dataset/train/ , pretrain = None
        super.__init__()

    def build_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 학습 환경 설정
        model = models.resnet50(pretrained=True) # true 옵션으로 사전 학습된 모델을 로드

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
        model = model.to(device)

    def forward(self, x):
        super.forward(self, x)

    def train_model(self, config):
        super.train_model(self,config)

    def eval_model(self):
        super.eval_model()