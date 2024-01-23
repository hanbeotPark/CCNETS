'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
from .explainer import Explainer
from .reasoner import Reasoner
from .producer import Producer
from ..utils.setting import set_train_param_args

class ResidualNet():
    def __init__(self, args, explainer_net, reasoner_net):
        set_train_param_args(self, args)
        self.model_names = ["explainer", "reasoner"]
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.ngpu = args.ngpu
        self.explainer =  Explainer.create(args, explainer_net).to(self.device)
        self.reasoner =  Reasoner.create(args, reasoner_net).to(self.device)
        self._models = [self.explainer, self.reasoner]

    def get_models(self):
        return self._models

    def reason(self, x, e):
        with torch.no_grad():
            y = self.reasoner(x, e)
        return y

    def infer(self, x):
        with torch.no_grad():
            e = self.explainer(x)
            y = self.reasoner(x, e)
        return y
