import os
import fashionVCpredict_models as model
import torch.nn.functional as F
import torch.utils.data as data
import pickle
import argparse
from utils.util import *
from torchvision.transforms import transforms
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--top_img_path', type=str, default='D:/project/fashionVC/top', help='dataset path')
parser.add_argument('--bottom_img_path', type=str, default='D:/project/fashionVC/bottom', help='dataset path')
parser.add_argument('--ckpt_path', type=str, default='D:/project/sigir2019/checkpoint/', help='checkpoint path')
parser.add_argument('--category_path', type=str, default='D:/project/sigir2019/data/', help='meta information path')
parser.add_argument('--output_path', type=str, default='D:/project/sigir2019/data/', help='output path')
opt = parser.parse_args()

class fashionVC(data.Dataset):
    def __init__(self, root=None, input_path=None, adj_path=None):
        self.item_images = root
        self.input_path = input_path
        self.adj_path = adj_path
        self.set = set

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            Warp(512),
            transforms.ToTensor(),
            normalize,
        ])

        self.images = self.read_image_paths()

        with open(self.input_path, 'rb') as f:
            self.inp = pickle.load(f)

        print('number of images=%d' % len(self.images))

    def __getitem__(self, index):
        try:
            path = self.images[index]
            img = Image.open(path).convert('RGB')
        except:
            index = 0
            path = self.images[index]
            img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)

    def read_image_paths(self):
        img_paths = os.listdir(self.item_images)
        img_paths = [os.path.join(self.item_images, x) for x in img_paths]
        return img_paths

def predict(images, model, input):
    feature_var = torch.Tensor(images).float()  # nx3x512x512
    inp_var = torch.Tensor(input).float().unsqueeze(0).detach()  # 303x300
    output = model(feature_var, inp_var)

    for idx, attr in enumerate(labels_per_task):
        output[:, attr] = F.softmax(output[:, attr], dim=1)
    output = output.cpu().detach().numpy()
    return output


if __name__ == "__main__":
    # load model
    print('checkpoint path: {}'.format(opt.ckpt_path))
    model_path = opt.ckpt_path + 'deepfashion/model_best_91.4221.pth.tar'
    adj_path = opt.ckpt_path + 'deepfashion/deepfashion_adj.pkl'
    word2vec_path = opt.ckpt_path + 'deepfashion/deepfashion_glove_word2vec.pkl'
    with open(word2vec_path, 'rb') as f:
        inp = pickle.load(f)
    model = model.gcn_resnet101(num_classes=303, t=0.2, adj_file=adj_path)

    attribute_label_predicted = {}
    if os.path.isfile(model_path):
        print('loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict({k.replace('module.',''): v for k, v in
                                checkpoint['state_dict'].items()}, False)

        # get image labels in test
        predicted_attribute_feature = {}
        img_paths = [opt.top_img_path, opt.bottom_img_path]
        for img_path in img_paths:
            print('image path: {}'.format(img_path))
            img_files = os.listdir(img_path)
            item_id = [x[:-4] + '\n' for x in img_files]
            with open(opt.category_path+'item_id.txt', 'a') as f:
                f.writelines(item_id)
            print('item_id.txt saved number: {}'.format(len(item_id)))

            model.eval()
            test_dataset = fashionVC(root=img_path, input_path=word2vec_path, adj_path=adj_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            test_loader = tqdm(test_loader)

            for idx, images in enumerate(test_loader):
                result = predict(images, model=model, input=inp)
                predicted_attribute_feature[item_id[idx].replace('\n', '')] = result[0]
        with open(opt.output_path+'predicted_attribute_feature.pkl', 'wb') as f:
            pickle.dump(predicted_attribute_feature, f)

    else:
        print('no checkpoint found at {}'.format(model_path))