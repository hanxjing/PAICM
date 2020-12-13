import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from time import time
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

abs_path = 'D:/project/sigir2019/'
# abs_path = '/home/hanxianjing/sigir2019/'

train_data_path = abs_path+'data/train_feature.pkl'
valid_data_path = abs_path+'data/valid_feature.pkl'
test_data_path = abs_path+'data/test_feature.pkl'
sampled_id_path = abs_path+'data/'
ckpt_save_path = abs_path+'checkpoint/paicm/'
ckpt_load_path = abs_path+'checkpoint/paicm/cls_e29'

test_with_checkpoint = False    # if true, load the saved checkpoint and then test

batch_size = 64
input_dim = 362
out_dim = 2
hidden_dim = 256
style_num = 60
outfit_num = 49710
n_epochs = 30

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.i_feature, self.j_feature, self.k_feature = pickle.load(f)
        self.feature_num = len(self.i_feature)

    def __getitem__(self, index):
        return self.i_feature[index], self.j_feature[index], self.k_feature[index]

    def __len__(self):
        return self.feature_num

class PAICM(nn.Module):
    def __init__(self, matirx_pos, matirx_neg):
        super(PAICM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.style_num = style_num
        self.outfit_num = outfit_num

        self.matirx_pos = matirx_pos
        self.matirx_neg = matirx_neg

        self.top_emb = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid())
        self.bottom_emb = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid())

        self.P_nmf = Parameter(torch.randn([2 * self.input_dim, self.style_num]).float().cuda())
        self.N_nmf = Parameter(torch.randn([2 * self.input_dim, self.style_num]).float().cuda())
        self.H1_nmf = Parameter(torch.randn([self.style_num, self.outfit_num]).float().cuda())
        self.H2_nmf = Parameter(torch.randn([self.style_num, self.outfit_num]).float().cuda())

    def forward(self, i_feature, j_feature, k_feature):
        ij_score = torch.matmul(self.top_emb(i_feature), torch.transpose(self.bottom_emb(j_feature), 0, 1))
        ik_score = torch.matmul(self.top_emb(i_feature), torch.transpose(self.bottom_emb(k_feature), 0, 1))
        ij_score = torch.diag(ij_score)
        ik_score = torch.diag(ik_score)
        
        self.P_nmf.data = torch.relu_(self.P_nmf.data)
        self.N_nmf.data = torch.relu_(self.N_nmf.data)
        self.H1_nmf.data = torch.relu_(self.H1_nmf.data)
        self.H2_nmf.data = torch.relu_(self.H2_nmf.data)

        style_ij_score = self.get_style_score(i_feature, j_feature, self.P_nmf)
        style_ik_score = self.get_style_score(i_feature, k_feature, self.N_nmf)

        ijk = torch.sigmoid(ij_score - ik_score)
        style_ijk = torch.sigmoid(style_ij_score - style_ik_score)

        loss1 = -torch.mean(torch.log(ijk))
        loss2 = -torch.mean(torch.log(style_ijk))
        loss3 = torch.mean(torch.square(self.matirx_pos - torch.matmul(self.P_nmf, self.H1_nmf)) +
                           (torch.square(self.matirx_neg - torch.matmul(self.N_nmf, self.H2_nmf))))
        loss = loss1 + loss2 + 0.1 * loss3
        ij_ik_score = torch.cat([ij_score.view(-1, 1), ik_score.view(-1, 1)], dim=1)
        auc = torch.mean((torch.argmax(ij_ik_score, dim=1) == 0).float())
        return loss, loss1, loss2, loss3, auc

    def get_style_score(self, i_feature, j_feature, styles):
        outift = torch.cat([i_feature, j_feature], dim=1)
        style_score = torch.zeros([i_feature.shape[0]])
        for i in range(outift.shape[0]):
            style_index = torch.argmin(torch.mean(torch.square(outift[i, :].view(-1, 1)-styles), dim=0))
            style = styles[:, style_index].view(2, self.input_dim)
            style_score[i] = torch.matmul(self.top_emb(style[0, :]), self.bottom_emb(style[1, :]))
        return style_score

def train(model, train_loader, valid_loader, optimizer, epoch=None, f=None):

    model.train()
    train_loss = []
    train_auc = []
    train_loss1 = []
    train_loss2 = []
    train_loss3 = []

    print("Epoch  {}/{}".format(epoch, n_epochs))
    start_time = time()
    for data in tqdm(train_loader):
        i_feature, j_feature, k_feature = data
        i_feature, j_feature, k_feature = i_feature.float().cuda(), \
                                          j_feature.float().cuda(), \
                                          k_feature.float().cuda()
        loss, loss1, loss2, loss3, auc = model(i_feature, j_feature, k_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())
        train_loss3.append(loss3.item())
        train_auc.append(auc.item())

    print("epoch {}: Loss {:.4f}, Loss1 {:.4f}, Loss2 {:.4f}, Loss3 {:.4f}  Train auc is:{:.4f}".format(epoch, np.mean(train_loss), np.mean(train_loss1), np.mean(train_loss2),
                                                                                              np.mean(train_loss3), np.mean(train_auc)))
    f.write("epoch {}: Loss {:.4f}, Loss1 {:.4f}, Loss2 {:.4f}, Loss3 {:.4f}  Train auc is:{:.4f}".format(epoch, np.mean(train_loss), np.mean(train_loss1), np.mean(train_loss2),
                                                                                              np.mean(train_loss3), np.mean(train_auc)))
    f.flush()
    end_time = time()
    print('Train duration: {}'.format(end_time-start_time))

    # validation
    valid_auc = []
    model.eval()
    for data in tqdm(valid_loader):
        i_feature, j_feature, k_feature = data
        i_feature, j_feature, k_feature = i_feature.float().cuda(), \
                                          j_feature.float().cuda(), \
                                          k_feature.float().cuda()
        loss, loss1, loss2, loss3, auc = model(i_feature, j_feature, k_feature)

        valid_auc.append(auc.item())

    print("epoch {}: Valid auc is:{:.4f}".format(epoch, np.mean(valid_auc)))
    f.write("epoch {}: Valid auc is:{:.4f}".format(epoch, np.mean(valid_auc)))
    f.flush()

    return np.mean(valid_auc)

def test(model, test_loader):

    model.eval()
    test_auc = []

    for data in tqdm(test_loader):
        i_feature, j_feature, k_feature = data
        i_feature, j_feature, k_feature = i_feature.float().cuda(), \
                                          j_feature.float().cuda(), \
                                          k_feature.float().cuda()
        loss, loss1, loss2, loss3, auc = model(i_feature, j_feature, k_feature)

        test_auc.append(auc.item())

    return np.mean(test_auc)

def main():

    train_data = Dataset(train_data_path)
    valid_data = Dataset(valid_data_path)
    test_data = Dataset(test_data_path)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False)

    # generate matrix
    print('generate matrix')
    with open(train_data_path, 'rb') as f:
        i_features, j_features, k_features = np.asarray(pickle.load(f), dtype='float32')
    matirx_pos = np.transpose(np.concatenate([i_features, j_features], axis=1))[:, 0:outfit_num]
    matirx_neg = np.transpose(np.concatenate([i_features, k_features], axis=1))[:, 0:outfit_num]
    matirx_pos, matirx_neg = Variable(torch.from_numpy(matirx_pos).float().cuda()), Variable(
        torch.from_numpy(matirx_neg).float().cuda())
    print('number of training sample {}'.format(len(i_features)))

    model = PAICM(matirx_pos, matirx_neg).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    if test_with_checkpoint:
        checkpoint = torch.load(ckpt_load_path)
        model.load_state_dict(checkpoint)
        test_auc = test(model, test_loader)
        print('test auc is {}'.format(test_auc))
        return

    print('start training')
    f = open('log.txt', 'a')
    f.write('*********************************\n')
    best_val = 0.
    best_test = 0.
    best_test_epoch = 0

    for epoch in range(0, n_epochs):
        valid_auc = train(model, train_loader, valid_loader, optimizer, epoch, f)
        test_auc = test(model, test_loader)
        print("epoch {}: Test auc is:{:.4f}".format(epoch, test_auc))
        f.write("epoch {}: Test auc is:{:.4f}".format(epoch, test_auc))
        f.flush()
        print("-" * 16)

        if valid_auc > best_val:
            best_test = test_auc
            best_test_epoch = epoch

            if best_test > 0.71:
                torch.save(model.state_dict(), ckpt_save_path + 'cls_e{}'.format(epoch))
                print('model saved')

    f.close()
    print('best test {} in epoch {}'.format(best_test, best_test_epoch))

if __name__ == '__main__':
    main()
