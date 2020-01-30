import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mmcv.cnn import normal_init

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


@BACKBONES.register_module
class ConvNet(nn.Module):
    """ Simple convolutional backbone with convolutional and max-pooling layers.
    """
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels):
        super(ConvNet, self).__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=5)
        
        self.conv_hidden = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=5)
        
        self.conv_out = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3)
        
        self.i = 0
    

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from mmdet.apis import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m)
                #elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                #    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        
        # Debug
        #plt.imshow(x.clone().cpu().numpy()[0, 0], cmap='hot')
        #plt.savefig("%i.png" % self.i)
        #self.i += 1
        
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv_in(x))
        #x = F.max_pool2d(F.relu(self.conv_hidden(x)), (2, 2))
        x = F.relu(self.conv_hidden(x))
        x = F.relu(self.conv_hidden(x))
        x = F.relu(self.conv_hidden(x))
        x = F.relu(self.conv_out(x))

        
        # The framework assumes tuple output because usually several last layers are out putted.
        # This way dimensions match again
        return [x]