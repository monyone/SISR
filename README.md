# Single Image Super Resolution (SISR)

## Requirement

* python3
* pytorch
* torchvision

## Models

* [SRCNN (ECCV 2014)](./models/SRCNN/) [[Paper]](https://arxiv.org/abs/1501.00092)
* [FSRCNN (ECCV 2016)](./models/FSRCNN/) [[Paper]](https://arxiv.org/abs/1608.00367)
* [ESPCN (CVPR 2016)](./models/ESPCN/) [[Paper]](https://arxiv.org/abs/1609.05158)
* [VDSR (CVPR 2016)](./models/VDSR/) [[Paper]](https://arxiv.org/abs/1511.04587)
* [DRCN (CVPR Oral 2016)](./models/DRCN/) [[Paper]](https://arxiv.org/abs/1511.04491)
* [RED-Net (CVPR 2016)](./models/REDNet/) [[Paper]](https://arxiv.org/abs/1603.09056)
* [DRRN (CVPR 2017)](./models/DRRN/) [[Paper]](https://ieeexplore.ieee.org/document/8099781)
* [LapSRN (CVPR 2017)](./models/LapSRN/) [[Paper]](https://arxiv.org/abs/1704.03915)
* [SRResNet, SRGAN (CVPR Oral 2017)](./models/SRGAN/) [[Paper]](https://arxiv.org/abs/1609.04802)
* [EDSR (CVPR Workshop 2017)](./models/EDSR/) [[paper]](https://arxiv.org/abs/1707.02921)
* [EnhanceNet (ICCV 2017)](./models/EnhanceNet/) [[Paper]](https://arxiv.org/abs/1612.07919)
* [MSLapSRN (TPAMI 2018)](./models/MSLapSRN/) [[Paper]](https://arxiv.org/abs/1710.01992)
* [TSRN (ECCV 2018)](./models/TSRN/) [[Paper]](https://arxiv.org/abs/1808.00043)
* [ESRNet, ESRGAN (ECCV 2018)](./models/ESRGAN/) [[Paper]](https://arxiv.org/abs/1809.00219)

## Usage

### Train

#### Geneator

```bash
./train.py --generator [ONE_OF_ABOVE_MODELS] --epoch ${epochs} --scale ${upscale_factor}
```

### GAN

```bash
./train.py --generator [ONE_OF_ABOVE_MODELS] --discriminator [ONE_OF_ABOVE_DISCRIMINATIOR] --epoch ${epochs} --scale ${upscale_factor}
```


### Super Resolution

```bash
./main.py --model [ONE_OF_ABOVE_MODELS] --image ${image_path} --state ${state_path} --scale ${upscale_factor}
```
