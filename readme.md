

envirnment 

    conda install python=3.7 cudatoolkit=10 cudnn pytorch torchvision
    conda install -c anaconda django 
    conda install -c conda-forge django-bootstrap3=10.0 django-jquery 
    pip install -r requirement.txt
    
start web demo:

    sh set_up.sh 
    ##make sure CUDA_VISIBLE_DEVICES is exsisting if there is a multi gpu envirnemnt, default is 0