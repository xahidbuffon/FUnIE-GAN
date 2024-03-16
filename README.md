This is the reproduction project for the course Deep Learning at TU Delft.

Goal: reproduce https://www.sciencedirect.com/science/article/pii/S1047320323002717?ref=pdf_download&fr=RR-2&rr=8509983ebc1eb760#da1

## Branch Rule

- `master`: the original content of the forked project
- `sea-pix-gan`: result of reproduction

for each feature, please work on seperate branches before merging to `sea-pix-gan`

## TODO & File structure

```
Sea-Pix-GAN-reproduction
    |- data
    |- Evaluation
    |- Pytorch
    |   |- models                       # TODO: put trained models here
    |   |- nets
    |   |   |- seapixgan
    |   |   |   |- `seapixgan.py`       # TODO: impl full network
    |   |   |   |- `generator.py`       # TODO: impl generator
    |   |   |   |- `discriminator.py`   # TODO: impl discriminator
    |   |- `train_seapixgan.py`         # TODO: traing script
    |   |- `test.py`                    # TODO: modify for our own test
```