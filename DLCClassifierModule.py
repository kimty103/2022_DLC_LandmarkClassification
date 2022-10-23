from ExampleClassifierModule import ExampleClassifier


class DLCClassifierModule(ExampleClassifier):
    def __init__(self, path_data, pretrain=None):   # path data = dataset/train/ , pretrain = None
        super.__init__()

    def build_model(self):
        super.build_model(self) #hello world

    def forward(self, x):
        super.forward(self, x)

    def train_model(self, config):
        super.train_model(self,config)

    def eval_model(self):
        super.eval_model()