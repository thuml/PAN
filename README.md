# PAN

Code release for ["Progressive Adversarial Networks for Fine-Grained Domain Adaptation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Progressive_Adversarial_Networks_for_Fine-Grained_Domain_Adaptation_CVPR_2020_paper.pdf) (CVPR 2020)

## Prerequisites:

* Python3
* PyTorch == 0.4.1 (with suitable CUDA and CuDNN version)
* torchvision >= 0.2.1

## Dataset:

You need to modify the path of the image in every ".txt" in "./dataset_list".

The sub-dataset CUB-200-Paintings of CUB-Paintings is provided in the following Google Drive links:
https://drive.google.com/file/d/1G327KsD93eyGTjMmByuVy9sk4tlEOyK3/view?usp=sharing


## Training on one dataset:

You can use the following commands to the tasks:

python PAN.py --gpu_id n --source c --target p

## Citation:

If you use this code for your research, please consider citing:

```
@inproceedings{PAN_20,
  title={Progressive Adversarial Networks for Fine-Grained Domain Adaptation},  
  author={Wang, Sinan and Chen, Xinyang and Wang, Yunbo and Long, Mingsheng and Wang, Jianmin}, 
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  pages={9213-9222}, 
  year={2020} 
}
```
## Contact
If you have any problem about our code, feel free to contact thusinan@foxmail.com.
