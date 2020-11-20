# Prototype-guided Attribute-wise Interpretable Scheme for Clothing Matching

Code for paper [Prototype-guided Attribute-wise Interpretable Scheme for Clothing Matching](https://dl.acm.org/doi/10.1145/3331184.3331245).

## Dependencies

This project is implemented with

- Python 3.8

-[Pytorch](pytorch.org) 1.6.0


## Data Preparation

First, you need to download the origin [FashionVC](https://xuemengsong.github.io/) data set.

(Google Drive Link: https://drive.google.com/open?id=1lO7M-jSWb25yucaW2Jj-9j_c9NqquSVF
Baidu Netdisk Link: https://pan.baidu.com/s/1eS1vNNk with the password: ytu4)

To extract the feature, we provided the item attribute classifier ([checkpoint](https://pan.baidu.com/s/1EbmJIYosNVyQoBk-NNKX_Q) password: k29k) pre-trained on DeepFashion dataset to generate the attribute feature.

Then we extracted the categoty and color labels from the meta textual and visual data of each item. The extracted categoty and color labels are provided in ./data/. The extraction tools are also provided in ./utils/.

### /instruction

python fashionVCpredict.py

python concat_category_color.py

### /data



### Meta data

format: user/outfit/item

Can be download from [there](https://drive.google.com/open?id=1sTfUoNPid9zG_MgV--lWZTBP1XZpmcK8).

## Running command

CUDA_VISIBLE_DEVICE=0 python main.py

## Citations

```
@inproceedings{song2019gp,
  title={GP-BPR: Personalized Compatibility Modeling for Clothing Matching},
  author={Song, Xuemeng and Han, Xianjing and Li, Yunkai and Chen, Jingyuan and Xu, Xin-Shun and Nie, Liqiang},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={320--328},
  year={2019}
}
```
