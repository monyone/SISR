# Single Image Super Resolution (SISR)

## Requirement

* python3
* pytorch
* torchvision

## Models

* [SRCNN (ECCV 2014)](./models/SRCNN/)
* [VDSR (CVPR 2016)](./models/VDSR/)
* [FSRCNN (ECCV 2016)](./models/FSRCNN/)
* [DRCN (CVPR 2016)](./models/DRCR/)
* [ESPCN (CVPR 2016)](./models/ESPCN/)
* [SRResNet (CVPR 2016)](./models/SRResNet/)

## Usage

### Train

```bash
./training.py --model [ONE_OF_ABOVE_MODELS] --epoch ${epochs}
```

### Super Resolution

```bash
./main.py --model [ONE_OF_ABOVE_MODELS] --image ${image_path} --state ${state_path} --scale ${upscale_size}
```
