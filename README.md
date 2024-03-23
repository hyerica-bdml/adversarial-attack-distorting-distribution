# Adversarial Attacks on Medical Segmentation Model via Transformation of Feature Statistics

Official Pytorch code for ["Adversarial Attacks on Medical Segmentation Model via Transformation of Feature Statistics"]() 

<img src="./figs/case_study.png" width="600">

## Introduction:
**Motivation:** Deep learning-based segmentation models, particularly those utilizing U-Net architectures, have significantly advanced medical imaging procedures, especially in CT image segmentation. Despite their impressive performance, these models remain susceptible to adversarial attacks. Traditional adversarial strategies often involve adding noise or subtle perturbations to the images, which can affect the balance between the attack's success rate and its detectability by human observers.

**Method:** In response to these challenges, this study introduces a new genre of adversarial attacks designed to mislead both the target segmentation model and medical professionals. The proposed method manipulates the texture statistics of an organ while preserving its original shape, leveraging a real-time style transfer technique known as the texture reformer. This approach is centered around a modified version of Adaptive Instance Normalization (AdaIN), traditionally used to align the feature statistics between source and target images, to subtly alter the organ's appearance in the CT images without making the changes perceptible.

**Results:** Extensive experimental validation confirms the efficacy of our method. The generated adversarial samples convincingly mimic realistic images in blind tests with medical practitioners, outperforming existing adversarial techniques. This novel approach not only provides a powerful benchmarking tool for evaluating the resilience of automated CT segmentation systems but also introduces a valuable method for data augmentation, thus improving the models' ability to generalize. Such dual functionality marks a significant contribution to the advancement of deep learning applications in medical and healthcare segmentation models.

**Contact:** nongaussian@hanyang.ac.kr.


## Environment:
- Python 3.9
- Pytorch 2.0.1
- Torchvision 0.15.2
- Pillow 9.3.0
- Numpy 1.24.1
- opencv-python 4.8.0

## Getting Started:
#### Step 1: Clone this repo

`git clone https://github.com/hyerica-bdml/adversarial-attack-transformation-statistics`  
`cd adversarial-attack-transformation-statistics`

#### Step 2: Prepare models

- Download the pre-trained auto-encoder models from this [google drive](). Unzip and place them at path `weights/`.

#### Step 3: Run transfer script

- For transforming statistics of features, you only need to input two images: the input image -image, the organ label of input image -label, like follows:
`python main.py -image inputs/image.npy -label inputs/label.npy`


## Script Parameters:
Specify inputs and outputs

- `-imgf` : File path to the image.
- `-lblf` : File path to the label.
- `-outf` : Folder to save output images.

Runtime controls

- `-coarse_alpha` : Hyperparameter to blend transformed feature with content feature in coarse level (level 5).
- `-fine_alpha` : Hyperparameter to blend transformed feature with content feature in fine level (level 4).
- `-concat_weight` : Hyperparameter to control the semantic guidance/awareness weight for -semantic concat mode and -semantic concat_ds mode, range 0-inf.
- `-coarse_psize` : Patch size in coarse level (level 5), 0 means using global view.
- `-fine_psize` : Patch size in fine level (level 4).
- `-enhance_alpha` : Hyperparameter to control the enhancement degree in level 3, level 2, and level 1.
- `-noise_mu` : Hyperparameter to control the noise rate of mean in AdaIN.
- `-noise_sigma` : Hyperparameter to control the noise rate of std in AdaIN.

## Citation:
If you find this code useful for your research, please cite the paper:
```
@article{lee2024adversarial,
  title={Adversarial Attacks on Medical Segmentation Model via Transformation of Feature Statistics},
  author={Lee, Woonghee and Ju, Mingeon and Sim, Yura and Jung, Young Kul and Kim, Tae Hyung and Kim, Younghoon},
  journal={Applied Sciences},
  volume={14},
  number={6},
  pages={2576},
  year={2024},
  publisher={MDPI}
}
```

## Acknowledgement:
We refer to python codes from [Texture-Reformer](https://github.com/EndyWon/Texture-Reformer) and [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation). Great thanks to them!
Inputs are taken from [Multi-Atlas Labeling Beyond the Cranial Vault](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).
