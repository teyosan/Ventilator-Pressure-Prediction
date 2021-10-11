# Python Libraries
import math

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Pytorch Image Models
import timm

# --------------------------------------
# Arc face
# --------------------------------------
class ArcMarginProduct(nn.Module):
    """
    bestfitting氏のarcmarginproduct2
    label smoothingが入っている
    https://github.com/bestfitting/instance_level_recognition/blob/683f021b4e65876835f028797ec28b0d1071bb45/src/layers/metric_learning.py#L66
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label == None:
            return cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine.float() > 0, phi, cosine.float())
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# --------------------------------------
# GeM Pooling layers
# --------------------------------------
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class Metric_model(nn.Module):
    def __init__(self, args):
        super(Metric_model, self).__init__()

        self.args = args
        self.backbone = timm.create_model(args.model_name, num_classes=0, pretrained=args.pretrained)
        self.global_pool = gem
        self.embedding_size = args.embedding_size

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if args.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.num_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif args.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.num_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.num_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )
        print(f's:{args.arcface_s},m:{args.arcface_m}')
        self.face_margin_product = ArcMarginProduct(self.embedding_size,
                                                    args.target_size,
                                                    s=args.arcface_s,
                                                    m=args.arcface_m)

    def extract_feature(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # チャネルごとの代表値
        x = self.neck(x)
        return x

    def forward(self, ｘ, label, **kargs):
        feature = self.extract_feature(x)
        out_face = self.face_margin_product(feature, label)

        return out_face, feature

class CFG:
    # model
    model_name = 'resnet18d'#'resnet18dtf_efficientnet_b7_ns
    pretrained = True
    # data
    target_size = 100
    # metric
    embedding_size = 512
    arcface_s = 30
    arcface_m = 0.3
    neck = 'option-D'

if __name__ == '__main__':
    model = Metric_model(args=CFG)
    print(model)
    batch_size = 4
    data = torch.randn(batch_size, 3, 224, 224, requires_grad=True, )
    label = torch.randint(CFG.target_size, (batch_size,), dtype=torch.int64)
    model.train()
    out = model(data, label)
    """
    loss = self.criterion(outputs, labels)
    if type(outputs) == tuple:
        logits = outputs[0]
    else:
        logits = outputs
    probs = F.softmax(logits).data
    train_acc = (probs.argmax(dim=1) == labels).float().mean()
    """