from utils.get_color import *
import pickle
import json

id_path = 'D:/project/sigir2019_d/data/item_id.txt'
attribute_feature_path = 'D:/project/sigir2019/data/'

item_category_path = 'D:/project/sigir2019_d/data/item_category.txt'
item_color_path = 'D:/project/sigir2019_d/data/item_color.txt'
sampled_id_path = 'D:/project/sigir2019_d/data/id/'
output_path = 'D:/project/sigir2019_d/data/'

# 生成包含所有category和color的文件
categories = set()
with open(item_category_path, 'r') as f:
    lines = f.readlines()
for line in lines:
    categories.add(line.split(maxsplit=1)[1])

colors = set()
with open(item_color_path, 'r') as f:
    lines = f.readlines()
for line in lines:
    colors.add(line.split(maxsplit=1)[1])

# 生成category对应的onehot向量
categories = [x[:-1] for x in categories]
category2vec = {}
vec_len = len(categories)
for i, category in enumerate(categories):
    category2vec[category] = np.zeros(vec_len)
    category2vec[category][i] = 1

id_category = {}
with open(item_category_path, 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.split(maxsplit=1)
    id_category[line[0]] = category2vec[line[1].replace('\n', '')]

# 生成color对应的onehot向量
colors = [x[:-1] for x in colors]
color2vec = {}
vec_len = len(colors)
for i, color in enumerate(colors):
    color2vec[color] = np.zeros(vec_len)
    color2vec[color][i] = 1

id_color = {}
with open(item_color_path, 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.split()
    id_color[line[0]] = color2vec[line[1]]

# 进行特征拼接
ids = np.loadtxt(id_path)
with open(attribute_feature_path+'predicted_attribute_feature.pkl', 'rb') as f:
    features = pickle.load(f)
new_features = {}
for i, id in enumerate(ids):
    new_features[int(id)] = np.concatenate((features[str(int(id))], id_category[str(int(id))], id_color[str(int(id))]))

# 生成训练数据
train_data_ijk_id = np.loadtxt(sampled_id_path+'train_ijk_shuffled_811.txt', delimiter="\t", dtype=int)
valid_data_ijk_id = np.loadtxt(sampled_id_path+'valid_ijk_shuffled_811.txt', delimiter="\t", dtype=int)
test_data_ijk_id = np.loadtxt(sampled_id_path+'test_ijk_shuffled_811.txt', delimiter="\t", dtype=int)
print(len(train_data_ijk_id))
label = ['train', 'valid', 'test']
ijk_id = [train_data_ijk_id, valid_data_ijk_id, test_data_ijk_id]
for i in range(3):
    feature_i = []
    feature_j = []
    feature_k = []
    for j in range(len(ijk_id[i])):
        feature_i.append(new_features[ijk_id[i][j][0]])
        feature_j.append(new_features[ijk_id[i][j][1]])
        feature_k.append(new_features[ijk_id[i][j][2]])
    with open(output_path+label[i]+'_feature.pkl', 'wb') as f:
        pickle.dump([feature_i, feature_j, feature_k], f)