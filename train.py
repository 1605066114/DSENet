from model.dsenet_model  import Dsenet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter


def train_net(net, device, data_path, epochs=1000, batch_size=4, lr=0.0001):
    # Load training set
    iter_num=0
    writer = SummaryWriter('E:/DSEnet/log/')
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # criterion = nn.MSELoss()

    best_loss = float('inf')
    # Training epoch times
    for epoch in range(epochs):
        # training model
        net.train()
        # Follow batch_size training
        for image, label in train_loader:
            iter_num=iter_num+1
            optimizer.zero_grad()
            # Copy data to the device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)


            # Using the network parameters, output the prediction results
            pred= net(image)
            # calculate loss


            if __name__ == '__main__':
                #DSENet The loss presented in the dsenet paper
                loss1 = 100*torch.mean(torch.abs((pred-label)/label))
                # loss1=criterion(label,pred)
                # loss2=criterion(m,label[:,:,256:256 + a, :])
                loss=loss1
            print('Loss/train', loss.item())


            image1=image[0, :, :]
            image1= (image1 - image1.min()) / (image1.max() - image1.min())
            label1 = label[0, :, :]
            label1 = (label1 - label1.min()) / (label1.max() - label1.min())
            pred1= pred[0, :, :]
            pred1 = (pred1 - pred1.min()) / (pred1.max() - pred1.min())
            writer.add_scalar('info/gloss', loss, iter_num)
            if iter_num % 20 == 0:

                writer.add_image('train/Image', image1, iter_num)
                writer.add_image('train/Prediction', pred1, iter_num)
                writer.add_image('train/GroundTruth', label1, iter_num)
            # Save the network parameter with the lowest loss value or every 50 epoches
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            if epoch >50 and epoch%50==0:
                torch.save(net.state_dict(), 'pth/epoch_' + str(epoch) + '.pth')
                print('save epoch', epoch)
            # Update parameters
            loss.backward()
            optimizer.step()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the network, the number of picture input channels is 1, the number of output channels is 1
    net = Dsenet(n_channels=1, n_classes=1)
    # net.load_state_dict(torch.load('pth/epoch_250.pth', map_location=device))
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)


    net.to(device=device)
    #
    data_path = "data/train/"
    train_net(net, device, data_path)