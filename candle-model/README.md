# VAE candle 

![](example.gif)

This gif was created by VAE model. The model was trained on 720x720 images of a candle flame. The model was trained on 128642images of size 256x256. The model was trained for 28 epochs. 

## Downloading the video
We will use video of a candle from [here](https://www.youtube.com/watch?v=HYjeos6ZXas). To download the video we could use [yt-dlp](https://github.com/yt-dlp/yt-dlp). Be aware that this is against YouTube's terms of service. Because we will be traininng the VAE on resolution smaller or same as 7p we don't have to waste our resources on higer resolution.
```bash
yt-dlp -S res:720 -o candle.webm "https://www.youtube.com/watch?v=HYjeos6ZXas"
```

## Extracting the frames
We will use [ffmpeg](https://ffmpeg.org/) to extract the frames from the video. We will extract 20 frames per second. 
```bash
mkdir -p frames/candle
ffmpeg -i candle.webm -vf fps=20 frames/candle/output_%06d.jpg
```

## Cropping the frames
Let's check how many frames we have here
```bash
ls frames/candle | wc -l # returns 128642
```
The candle flame is positioned in the centre so even if we will rescale the images there is no need for us to store the whole image. We will crop the images to 720x720 pixels.
```bash
cd frames/candle
find . -maxdepth 1 -type f -name '*.jpg' | parallel --bar -P 12 convert {} -gravity center -crop 720x720+0+0 +repage {} # this will take ca 1h
```
If you feel like removing images. Removing images will not work with `rm` because there are too many of them. Remove the whole folder instead.
```bash
cd ..
rm -rf frames
```

## Training model
```bash
python3.11 -m venv evn
source evn/bin/activate
python3.11 -m pip install -r requirements.txt
```
Then train the model in `vae_256.ipynb`. 

## Exporting the model
We need to export the model to ONNX format. We will use `export.py` for that. 
```bash
python3.11 export.py  
```

## pip freeze
```
appnope==0.1.3
asttokens==2.2.1
backcall==0.2.0
certifi==2023.5.7
charset-normalizer==3.2.0
comm==0.1.3
contourpy==1.1.0
cycler==0.11.0
debugpy==1.6.7
decorator==5.1.1
executing==1.2.0
filelock==3.12.2
fonttools==4.41.0
idna==3.4
imageio==2.31.1
ipykernel==6.24.0
ipython==8.14.0
jedi==0.18.2
Jinja2==3.1.2
jupyter_client==8.3.0
jupyter_core==5.3.1
kiwisolver==1.4.4
MarkupSafe==2.1.3
matplotlib==3.7.2
matplotlib-inline==0.1.6
mpmath==1.3.0
nest-asyncio==1.5.6
networkx==3.1
numpy==1.25.1
onnx==1.14.0
packaging==23.1
pandas==2.0.3
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==10.0.0
platformdirs==3.9.1
prompt-toolkit==3.0.39
protobuf==4.23.4
psutil==5.9.5
ptyprocess==0.7.0
pure-eval==0.2.2
Pygments==2.15.1
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
pyzmq==25.1.0
requests==2.31.0
scipy==1.11.1
six==1.16.0
stack-data==0.6.2
sympy==1.12
torch==2.0.1
torchsummary==1.5.1
torchvision==0.15.2
tornado==6.3.2
tqdm==4.65.0
traitlets==5.9.0
typing_extensions==4.7.1
tzdata==2023.3
urllib3==2.0.3
wcwidth==0.2.6
```