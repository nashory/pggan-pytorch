## Pytorch Implementation of "Progressive growing GAN (PGGAN)"
PyTorch implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)   
__YOUR CONTRIBUTION IS INVALUABLE FOR THIS PROJECT :)__ 

![image](https://puu.sh/ydG0E/e0f32b0d92.png)

## What's different from official paper?
+ to be updated...


## How to use?
__[step 1.] Prepare dataset__   
CelebA-HQ dataset is not available yet, so I used 100,000 generated PNGs of CelebA-HQ released by the author.   
The quality of the generated image was good enough for training and verifying the preformance of the code.  
If the CelebA-HQ dataset is releasted in then near future, I will update the experimental result.  
[[download]](https://drive.google.com/open?id=0B4qLcYyJmiz0MUVMVFEyclJnRmc)

+ CAUTION: loading 1024 x 1024 image and resizing every forward process makes training slow. I recommend you to use normal CelebA dataset until the output resolution converges to 256x256.

~~~
---------------------------------------------
The training data folder should look like : 
<train_data_root>
                |--classA
                        |--image1A
                        |--image2A ...
                |--classB
                        |--image1B
                        |--image2B ...
---------------------------------------------
~~~

__[step 2.] Prepare environment using virtualenv__   
  + you can easily set PyTorch and TensorFlow evnironment using virtualenv.  
  + CAUTION: if you have trouble installing PyTorch, install it mansually using pip. [[PyTorch Install]](http://pytorch.org/)
  
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
 
  
__[step 4.] Display on tensorboard__   
+ you can check the results on tensorboard.
  ~~~
  $ tensorboard --logdir runs --port 8888
  $ <host_ip>:8888 at your browser.
  ~~~
  
  
## Experimental results
to be updated ...

## To-Do List (will be implemented soon)
- [X] Equalized learning rate (weight normalization)
- [X] Support WGAN-GP loss


## Acknowledgement
+ [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)

## Author
MinchulShin, [@nashory](https://github.com/nashory)  
![image](https://camo.githubusercontent.com/e053bc3e1b63635239e8a44574e819e62ab3e3f4/687474703a2f2f692e67697068792e636f6d2f49634a366e36564a4e6a524e532e676966)


