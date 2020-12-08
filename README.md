# Ear detection using Viola-Jones algorithm

## Usage
```console
usage: main.py [-h] [-src SRC] [-dest DEST] [-annot ANNOT] [-plot]

optional arguments:
  -h, --help    show this help message and exit
  -src SRC      path to data file or directory.
  -dest DEST    path to destinatnion directory.
  -annot ANNOT  path to directory of annotated images.
  -plot         plot precision and recall curve.
```

## Viola-Jones parameters
These parameters were acquired by trying different values for scale step and size. 
Values tested for scale step were between 1.005 and 2.0 with step of 0.005 and tested values for
size were between 1 and 10 with step 1. Values were chosen based on highest average IOU (Intersection over Union) 
value.
```json
{
    "left_ear": {"scale_step": 1.0149, "size": 3},
    "right_ear": {"scale_step": 1.0199, "size": 1}
}
```