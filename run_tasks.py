import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data_augmentation import DA
from losses import FocalLoss, LossWrapper
from models import SelfAttenModel, LSTM_Model, ModalityFusionNet
import configs
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def label_binarize(labels):
    labels[labels == 1] = 0
    labels[labels > 1] = 1
    return labels

def load_data(filename=configs.DATA_PATH, job_type = '2-cla'):
    X_train = np.load(os.path.join(filename, 'X_train.npy'))
    X_test = np.load(os.path.join(filename, 'X_test.npy'))
    y_train = np.load(os.path.join(filename, 'y_train.npy'))
    y_test = np.load(os.path.join(filename, 'y_test.npy'))
    
    if job_type == "2-cla":
        y_train = label_binarize(y_train)
        y_test = label_binarize(y_test)
    
    return X_train, X_test, y_train, y_test

def load_data_all(filename=configs.DATA_PATH, job_type = 'cla2'):
    X = np.load(os.path.join(filename, 'features_ts.npy'))
    missing_ecg = np.load(os.path.join(filename, 'missing_ecg.npy'))
    missing_gsr = np.load(os.path.join(filename, 'missing_gsr.npy'))
    X = np.load(os.path.join(filename, 'features_ts.npy'))
    y = np.load(os.path.join(filename, 'labels_{}.npy'.format(job_type)))
    
    return X, missing_ecg, missing_gsr, y

def warp_loaders(X_train, y_train, missing_ecg_tensor, missing_gsr_tensor):
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    missing_ecg_tensor = torch.Tensor(missing_ecg_tensor)
    missing_gsr_tensor = torch.Tensor(missing_gsr_tensor)
    dataset = TensorDataset(X_train_tensor, y_train_tensor, missing_ecg_tensor, missing_gsr_tensor)
    data_loader = DataLoader(dataset, 
                             batch_size=configs.BATCH_SIZE, 
                             shuffle=True)
    return data_loader

def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
    loss_func = LossWrapper()
    
    for epoch in range(configs.EPOCH):
        for s, (batch_x, batch_y, batch_missing_ecg, batch_missing_gsr) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.unsqueeze(1).to(device)
            output_ecg, output_gsr, output_both = model(batch_x)
            # print(output)
            loss = loss_func(output_ecg, output_gsr, output_both, batch_y, batch_missing_ecg, batch_missing_gsr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch, '| training loss: %.4f' % loss.cpu().data.numpy())
        if (epoch + 1) % 10 == 0:
            save_path = configs.SAVE_PATH + 'checkpoint_{}.pth'.format(epoch + 1)
            torch.save(model.state_dict(), save_path)

def main():
    X, missing_ecg, missing_gsr, y = load_data_all(configs.DATA_PATH)
    print(X.shape)
    X_train, X_test, missing_ecg_train, missing_ecg_test, missing_gsr_train, missing_gsr_test, y_train, y_test = train_test_split(X, missing_ecg, missing_gsr, y, test_size=0.25)
    train_loader = warp_loaders(X_train, y_train, missing_ecg_train, missing_gsr_train)
    # X_train, y_train = DA(X_train, y_train, 5)
    # model = LSTM_Model().to(device)
    # model = SelfAttenModel(input_dim = X_train.shape[-1], 
    #                        embed_dim = 128,
    #                        hidden_dim = 256, 
    #                        output_dim = 1, 
    #                        dropout=0.5,
    #                        device=device,
    #                        sequence_length=X_train.shape[-2]).to(device)
    # summary(model, X_train.shape[1:])
    model = ModalityFusionNet(embed_dim=128, hidden_dim=256, output_dim=1)
    # AE_weights = torch.load('saved_model/pretrain/SA-AE/checkpoint_50.pth')
    # model.load_my_state_dict(AE_weights)
    train(model, train_loader)
    
if __name__ == '__main__':
    main()
