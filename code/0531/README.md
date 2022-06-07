## Mango Competition 
#### Requirements 
* python 3.8   
* torch 1.5.0  
* Pillow
* matplotlib
* torchvision
* numpy
* pandas
* torchsummary

#### Usage
1. Download masked mango images from [here](https://drive.google.com/drive/folders/1yPxJ0YtfCzTdn5Zg5KAewxtwmsnhZpXz?usp=sharing)
2. Follow the reference file structure below and place all your files accordingly.
3. Modify parameter dictionary for your training session. Values for unspecified keywords will be default values as follows:  
<pre>parameter = {  
    'feature_extract':True,  
    'model_name':'squeezenet',  
    'batch_size':32,  
    'lr':0.005,  
    'momentum':0,  
    'num_epochs':15,  
}</pre>
4. Enter the following script in your command.  
`python3 main.py`  

#### File Structure
Be careful with all the folder names.  

```
final
│
│   engine.py
│   main.py
│   MangoDataset.py
│   plot_utils.py
│   README.md
│
└───MangoData
│   │
│   └───Dev
│   │   │   00027.jpg
│   │   │   00033.jpg
│   │   │   ...
│   │
│   └───maskDev
│   │   │   mask00027.jpg
│   │   │   mask00033.jpg
│   │   │   ...
│   │
│   └───maskTrain
│   │   │   mask00002.jpg
│   │   │   mask00003.jpg
│   │   │   ...
│   │
│   └───Train
│   │   │   00002.jpg
│   │   │   00003.jpg
│   │   │   ...
│   │   dev.csv
│   │   train.csv
│   │   ...
│
└───model_dict
│   │   ...
│
└───result_pics
│   │   ...
│      

```
