import torch
from videomodels.i3d_resnet import *
from videomodels.action_recognition import _C as _C_action_recognition

class NormalizeLayer_video(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer_video, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))

    def forward(self, input: torch.tensor, y=None):
        video=input.clone()
        dtype = video.dtype
        mean = torch.as_tensor(self.means, dtype=dtype, device=video.device)
        std = torch.as_tensor(self.sds, dtype=dtype, device=video.device)
        video=video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return video


def get_model(cfg):
    """Returns a pre-defined model by name

    Returns
    -------
    The model.
    """
    name = cfg.CONFIG.MODEL.NAME.lower()
    _models={'i3d_nl5_resnet50_v1_kinetics400': i3d_nl5_resnet50_v1_kinetics400}
        
    net = _models[name](cfg)
    return net


def change_cfg(cfg, batch_size):
    # modify video paths and pretrain setting.
    cfg.CONFIG.MODEL.PRETRAINED = False
    cfg.CONFIG.VAL.BATCH_SIZE = batch_size
    return cfg

def get_cfg_custom(cfg_path, batch_size=16):
    cfg=_C_action_recognition.clone()
    cfg.merge_from_file(cfg_path)
    cfg = change_cfg(cfg, batch_size)
    return cfg



def load_classifier(ckpt_path, model_name):
    cfg_path = ckpt_path+'/'+model_name+'.yaml'
    ckpt_path = ckpt_path+'/'+model_name+'.pth'
    cfg = get_cfg_custom(cfg_path,1)
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()    
    normalize_layer = NormalizeLayer_video(means=[0.485, 0.456, 0.406], sds=[0.229, 0.224, 0.225])
    return torch.nn.Sequential(normalize_layer, model)

 