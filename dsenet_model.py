
import torch.nn.functional as F

from dsenet_parts import *

#Form each block into a DSENet as described in the paper
class Dsenet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Dsenet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = TripleConv(n_channels, 16) #384×256×16
        self.down1 = Down(16, 32)            #192×128×32
        self.down2 = Down(32, 64)            #96×64×64
        self.down3 = Down(64, 128)           #48×32×128
        self.down4 = Down(128, 256)          #24×16×256
        self.down5 = Down(256, 512)          #12×8×512
        self.down6 = Down(512, 1024)         #6×4×1024

        self.up1 = Up(1536, 512, bilinear)  #12×8×512
        self.up2 = Up(768, 256, bilinear)   #24×16×256
        self.up3 = Up(384, 128, bilinear)   #48×32×128
        self.up4 = Up(192, 64, bilinear)    #96×64×64
        self.up5 = Up(96, 32, bilinear)     #192×128×32
        self.up6 = Up(48, 16, bilinear)     #384×256×16
        self.outc = OutConv(16, n_classes)  #384×256×1

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = Dsenet(n_channels=3, n_classes=1)
    print(net)
