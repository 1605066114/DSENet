import glob
import numpy as np
import torch
import os
import cv2
from model.dsenet_model  import Dsenet
from dsenet_model import  Dsenet
from skimage.metrics import structural_similarity
from sklearn import metrics
from scipy.ndimage.interpolation import zoom



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Dsenet(n_channels=1, n_classes=1)

    net.to(device=device)

    net.load_state_dict(torch.load('pth/epoch_950.pth', map_location=device))

    net.eval()

    tests_path = glob.glob('data/test/image/*.npy')

    for test_path in tests_path:

        save_res_path = 'predictions/'+test_path.split('.')[0] + '_res.npy'

        img = np.load(test_path)
        wid, high = img.shape
        img = zoom(img, (384 / wid, 256 / high), order=3) #Compressed resolution according to thesis requirements


        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        img_tensor = torch.from_numpy(img)

        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        pred = net(img_tensor)

        pred = np.array(pred.data.cpu()[0])[0]
        wid, high = pred.shape
        pred = zoom(pred, (512 / wid, 512 / high), order=3)
        save_res_path = 'predictions/result/' + test_path.split('.')[0].split('\\')[-1].split('_')[0] + '_' + str(int(test_path.split('.')[0].split('\\')[-1].split('_')[-1])) + '.npy'

        np.save(save_res_path, pred)
