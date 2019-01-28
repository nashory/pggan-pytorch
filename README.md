## Pytorch Implementation of "Progressive growing GAN (PGGAN)"
PyTorch implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)   
__YOUR CONTRIBUTION IS INVALUABLE FOR THIS PROJECT :)__ 

![image](https://puu.sh/ydG0E/e0f32b0d92.png)

## What's different from official paper?
+ original: trans(G)-->trans(D)-->stab / my code: trans(G)-->stab-->transition(D)-->stab
+ no use of NIN layer. The unnecessary layers (like low-resolution blocks) are automatically flushed out and grow.
+ used torch.utils.weight_norm for to_rgb_layer of generator.
+ No need to implement the the Celeb A data, Just come with your own dataset :)


## How to use?
__[step 1.] Prepare dataset__   
The author of progressive GAN released CelebA-HQ dataset, and which Nash is working on over on the branch that i forked this from. For my version just make sure that all images are the children of that folder that you declare in Config.py. Also i warn you that if you use multiple classes, they should be similar as to not end up with attrocities. 

~~~
---------------------------------------------
The training data folder should look like : 
<train_data_root>
                |--Your Folder
                        |--image 1
                        |--image 2
                        |--image 3 ...
---------------------------------------------
~~~

__[step 2.] Prepare environment using virtualenv__   
  + you can easily set PyTorch and TensorFlow evnironment using virtualenv.  
  + CAUTION: if you have trouble installing PyTorch, install it mansually using pip. [[PyTorch Install]](http://pytorch.org/)
  + For install please take your time and install all dependencies of PyTorch and also install tensorflow
  
  ~~~
  $ virtualenv --python=python2.7 venv
  $ . venv/bin/activate
  $ pip install -r requirements.txt
  ~~~



__[step 3.] Run training__      
+ edit `config.py` to change parameters. (don't forget to change path to training images)
+ specify which gpu devices to be used, and change "n_gpu" option in `config.py` to support Multi-GPU training.
+ run and enjoy!  

~~~~
  (example)
  If using Single-GPU (device_id = 0):
  $ vim config.py   -->   change "n_gpu=1"
  $ CUDA_VISIBLE_DEVICES=0 python trainer.py
  
  If using Multi-GPUs (device id = 1,3,7):
  $ vim config.py   -->   change "n_gpu=3"
  $ CUDA_VISIBLE_DEVICES=1,3,7 python trainer.py
~~~~
 
  
__[step 4.] Display on tensorboard__   (At the moment skip this part)
+ you can check the results on tensorboard.

<p align="center"><img src="https://puu.sh/ympU0/c38f4e7d33.png" width="700"></p>   
<p align="center"><img src="https://puu.sh/ympUe/bf9b53dea8.png" width="700" align="center"></p>   

  ~~~
  $ tensorboard --logdir repo/tensorboard --port 8888
  $ <host_ip>:8888 at your browser.
  ~~~
  
  
## Experimental results   
The model is still being trained at this moment.  
The result of higher resolution will be updated soon.  

__Generated Images__

<img src="https://github.com/nashory/gifs/blob/pggan-pytorch/scatch_4.gif?raw=true" width="430" height="430"> <img src="https://puu.sh/yufIa/2a56d61890.jpg" width="430" height="430">  
<img src="https://github.com/nashory/gifs/blob/pggan-pytorch/4_8.gif?raw=true" width="430" height="430"> <img src="https://puu.sh/yufJx/a427ccdcdf.jpg" width="430" height="430">  
<img src="https://github.com/nashory/gifs/blob/pggan-pytorch/8_16.gif?raw=true" width="430" height="430"> <img src="https://puu.sh/yufMz/dd74f56d36.jpg" width="430" height="430">  
<img src="https://github.com/nashory/gifs/blob/pggan-pytorch/16_32.gif?raw=true" width="430" height="430"> <img src="https://puu.sh/yufLF/013cc59c15.jpg" width="430" height="430">  
<img src="https://github.com/nashory/gifs/blob/pggan-pytorch/32_64.gif?raw=true" width="430" height="430"> <img src="https://puu.sh/yufMV/835ec431ea.jpg" width="430" height="430">  


__Loss Curve__

![image](https://puu.sh/yuhi4/a49686b220.png)

## To-Do List (will be implemented soon)
- Fix tensorboard support issues
- Add capability of supporting 1 channel images



## Compatability
+ cuda v8.0 (if you dont have it dont worry)
+ Tesla P40 (you may need more than 12GB Memory. If not, please adjust the batch_table in `dataloader.py`)


## Acknowledgement
+ [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)
+ [nashory/progressive-growing-torch](https://github.com/nashory/progressive-growing-torch)
+ [TuXiaokang/DCGAN.PyTorch](https://github.com/TuXiaokang/DCGAN.PyTorch)

##
## Author
MinchulShin, [@nashory](https://github.com/nashory)  
## Contributors
DeMarcus Edwards, [@Djmcflush](https://github.com/Djmcflush)
