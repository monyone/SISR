# Single Image Super Resolution (SISR)

## Requirement

* python3
* pytorch
* torchvision

## Models

* [SRCNN(2014)](./models/SRCNN/)
* [VDSR(2015)](./models/VDSR/)
* [FSRCNN(2016)](./models/FSRCNN/)

## Usage

### Train

```bash
./training.py --model [ONE_OF_ABOVE_MODELS] --epoch ${epochs}
```

### Super Resolution

```bash
./main.py --model [ONE_OF_ABOVE_MODELS] --image ${image_path} --state ${state_path} --scale ${upscale_size}
```
