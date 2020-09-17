import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SEMDataset
from models import UnsupNet
from losses import ReconstructionLoss
from utils.pretrain import train_test_split_sem


parser = argparse.ArgumentParser(description='Pre-train model on the SEM dataset.')
parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(256, 256), help='Image size to load for training.')
parser.add_argument('--save-dir', metavar='save_dir', type=str, default='./saved_models/', help='directory to save and load weights from.')
parser.add_argument('--batch-size', metavar='batch_size', type=int, default=10, help='Batch size for training.')
parser.add_argument('--lr', metavar='lr', type=float, default=3e-4, help='Learning rate.')
parser.add_argument('--epochs', metavar='epochs', type=int, default=50, help='No. of epochs to train.')
parser.add_argument('--load-ckpt', metavar='load_ckpt', type=str, default=None, help='model load path to resume training from.')
namespace = parser.parse_args()


train_dataset, val_dataset = train_test_split_sem(SEMDataset, 
                                                  namespace.data_dir, 
                                                  im_size=namespace.im_size, 
                                                  device=namespace.device)

print('Train: {}    Val: {}'.format(len(train_dataset), len(val_dataset)))
train_loader = DataLoader(train_dataset, batch_size=namespace.batch_size) 
val_loader = DataLoader(val_dataset, batch_size=namespace.batch_size)

model = UnsupNet().to(namespace.device)
if namespace.load_ckpt is not None:
    model.load_state_dict(torch.load(namespace.load_ckpt))
optimizer = Adam(model.parameters(), lr=namespace.lr)
criterion = ReconstructionLoss()

losses = {'train': [], 'val': []}

for epoch in range(namespace.epochs):

    epoch_train_losses = []
    start = time.time()
    model.train()
    for x in train_loader:
        z, x_prime = model(x)
        loss = criterion(x, x_prime)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())
    losses['train'].append(np.mean(epoch_train_losses))

    epoch_val_losses = []
    model.eval()
    for x in val_loader:
        z, x_prime = model(x)
        loss = criterion(x, x_prime)

        epoch_val_losses.append(loss.item())

    save_model = (np.mean(epoch_val_losses) < np.min(losses['val']) if epoch > 0 else False)
    losses['val'].append(np.mean(epoch_val_losses))

    print('{}/{}    Train: {:.5f}    Val: {:.5f}    T: {:.2f} s at {}'.format(epoch+1, namespace.epochs, losses['train'][-1], losses['val'][-1], time.time()-start, time.ctime()))

    if save_model:
        torch.save(model.encoder.state_dict(), '{}pretrained-encoder.pt'.format(namespace.save_dir))
        torch.save(model.decoder.state_dict(), '{}pretrained-encoder.pt'.format(namespace.save_dir))

with open('{}logs/pretrain-losses.pkl'.format(namespace.save_dir), 'wb') as f:
    pickle.dump(losses, f)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(losses['train'], label='train')
ax.plot(losses['val'], label='val')
ax.set_title('End Train: {:.5f}    End Val: {:.5f}'.format(losses['train'][-1], losses['val'][-1]))
ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Loss')

plt.savefig('{}logs/pretrain-losses.png'.format(namespace.save_dir), bbox_inches='tight', pad_inches=0.1)
plt.close()