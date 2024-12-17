# VideoPure

## Start
```
git clone https://github.com/deep-kaixun/VideoPure.git
cd VideoPure
pip install -r requirements.txt
```


## Pretrained Model(NL_res50)
Download [here](https://drive.google.com/file/d/19Xci1TRWBkBv7A7AiAw7tqTOnly-IQNG/view?usp=drive_link).

```bash
mv i3d_resnet50.pth ckpt/NL_res50.pth
```

## Run
```bash
python3 main.py --noise_type videopure

```

## Acknowledgements
[diffusers](https://github.com/huggingface/diffusers)





