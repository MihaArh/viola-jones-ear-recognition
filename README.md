# Ear detection using Viola-Jones algorithm

## Usage
```console
python main.py -p "{path_to_dir_or_file}"
```

## Viola-Jones parameters
These parameters were acquired by trying different values for scale step and size. 
Values tested for scale step were between 1.005 and 2.0 with step of 0.005 and tested values for
size were between 1 and 10 with step 1. Values were chosen based on average IOU (Intersection over Uniou) 
value.
```json
{
    "left_ear": {"scale_step": 1.0149, "size": 3},
    "right_ear": {"scale_step": 1.0199, "size": 1}
}
```