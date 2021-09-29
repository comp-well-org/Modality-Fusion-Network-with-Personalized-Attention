import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import configs
from models import SelfAttentionAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(configs.DATA_PATH + 'features_unlabel.npy')
print(data.shape)
data_tensor = torch.Tensor(data)
dataset = TensorDataset(data_tensor)
data_loader = DataLoader(dataset, 
                        batch_size=configs.BATCH_SIZE, 
                        shuffle=True)

model = SelfAttentionAE(data.shape[-1], 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
loss_func = nn.MSELoss()

for epoch in range(configs.EPOCH):
    for s, batch in enumerate(data_loader):
        batch = batch[0].to(device)
        output = model(batch, batch)
        # print(output)
        loss = loss_func(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch, '| training loss: %.4f' % loss.cpu().data.numpy())
    if (epoch + 1) % 10 == 0:
        save_path = configs.SAVE_PATH + 'checkpoint_{}.pth'.format(epoch + 1)
        torch.save(model.state_dict(), save_path)