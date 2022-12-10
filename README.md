# Adversarial Attack Medical Images by Distorting the Distribution of Features
**[update 12/10/2022]**

Official Pytorch code for ["Adversarial Attack Medical Images by Distorting the Distribution of Features"]() (? 2022)

## Introduction:



## Environment:
- Python 3.9
- Pytorch 1.11.0
- Torchvision 0.12.0
- Pillow 9.0.1
- Numpy 1.22.3
- opencv-python 4.6.0

## Getting Started:
#### Step 1: Clone this repo

`git clone https://github.com/hyerica-bdml/adversarial-attack-distorting-distribution`  
`cd adversarial-attack-distorting-distribution`

#### Step 2: Prepare models

- Download the pre-trained auto-encoder models from this [google drive](). Unzip and place them at path `weights/`.

#### Step 3: Run transfer script

- For distorting distribution, you only need to input two images: the input image -image, the organ label of input image -label, like follows:
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
@inproceedings{

}
```

## Acknowledgement:
We refer to python codes from [Texture-Reformer](https://github.com/EndyWon/Texture-Reformer) and [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation). Great thanks to them!
