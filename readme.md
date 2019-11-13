

envirnment 

    conda install python=3.7 cudatoolkit=10 cudnn pytorch torchvision
    conda install -c anaconda django 
    conda install -c conda-forge django-bootstrap3=10.0 django-jquery 
    pip install -r requirement.txt

model can be downloaded at [Google driver](https://drive.google.com/open?id=1R9wS8QJLp-f15chhOhNgpK7vAkCjtLDj).

start web demo:
    
    sh set_up.sh 
    ##make sure CUDA_VISIBLE_DEVICES is exsisting if there is a multi gpu envirnemnt, default is 0
    The position of model is search/model
