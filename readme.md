Co-segmentation network and web demo
========
This project is for the training of the co-segmentation network and a simple website of the project 

envirnment: 
all our envirnment is based on the Anaconda3, you can run code below in your anaconda envirnment directly

    conda install python=3.7 cudatoolkit=10 cudnn pytorch torchvision
    conda install -c anaconda django 
    conda install -c conda-forge django-bootstrap3=10.0 django-jquery 
    pip install -r requirement.txt
    pip install Jupyter

model can be downloaded at [Google driver](https://drive.google.com/open?id=1R9wS8QJLp-f15chhOhNgpK7vAkCjtLDj).
training dataset is based on the [Pascol VoC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

If using jupyter:
    
    jupyter notebook --no-browser --port=8889
    
    #ensure you do the dataset generation first
    #then use Jupyter

 dataset generation:
    
    #put the python files in co-seg/datasets into the data set postion
    #evaluation with next order
    python convert_val_anno.py
    #Convert channel 3 masks in SegmentationClass directory into channel 1 range [0,21] label and save them in label directory.
    
    python sperate_train_val.py
    #Sperate the labels into train and val according to the train.txt and val.txt
    
    python create_coseg_dataset.py
    #Put this script into a folder contains semantic label(e.g. label/train/). Extract all image pairs in current folder and create co-segmentation label saving it to colabel/train/

Model training:

    python co_seg/train1.py 
    #you may change the path in the train1.py 
    #path include:
    #    train_data_dir
    #    train_label_dir
    #    train_txt
    #    val_data_dir
    #    val_label_dir
    #    val_txt

ecaluation evaluation for the moedl:

    python co_seg/single_demo.py

start web demo in your anaconda envirnment:
    
    sh set_up.sh 
    ##make sure CUDA_VISIBLE_DEVICES is exsisting if there is a multi gpu envirnemnt, default is 0
    The position of model is search/model
