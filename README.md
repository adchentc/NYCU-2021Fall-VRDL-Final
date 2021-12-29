# NYCU_VRDL_Final_Project


## <div align="center">Introduction</div>

## <div align="center">Proposed Approach</div>
<p>
<!--    <a align="left" href="https://ultralytics.com/yolov5" target="_blank"> -->
   <img width="850" src="https://github.com/adchentc/NYCU_VRDL_Final_Project/blob/main/ourpropose-v1.png"></a>
</p>

## <div align="center">Usage</div>

### Dependencies
  - Python 3.8.8
  - PyTorch 1.9.0
  - ml_collections
  
#### 1. Object Detection
Please refer to Official Github for YOLOv5, see this [link](https://github.com/ultralytics/yolov5)
<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Train</summary>

 
 ```bash
 $ python train.py --img 640 --batch 6 --epochs 20 --data dataset.yaml --weights yolov5m.pt

```

</details>

<details open>
<summary>Inference</summary>

 
 ```bash
 $ python detect.py --source /testfolder --weights /train/exp/weights/best.pt --conf 0.1
```

#### 2. Classifier
1. Download Google Pre-trained ViT Models \
   The model can be downloaded in this [link](https://console.cloud.google.com/storage/browser/vit_models).
   file: vit_models/imagenet21k+imagenet2012/ViT-L_16.npz

2. Install required packages
      ```
      pip3 install -r requirements.txt
      ```
3. Train \
   Run: 
      ```
      bash train.sh
      ```
   by change some parameter (you can change the classifier by assign the 'classifier' : 'transfg' for TransFG and 'resnet50' for ResNet50)
      ```
     python -m torch.distributed.launch --nproc_per_node=1 train.py \
      --name 'taskname' \
      --dataset 'myFish' \
      --data_root 'pathto/training_images' \
      --model_type "ViT-L_16" \
      --pretrained_dir "vitpretrainedmodel_path/imagenet21k+imagenet2012_ViT-L_16.npz" \
      --output_dir "outputdirectory" \
      --train_batch_size 2 \
      --eval_batch_size 2 \
      --local_rank 0 \
      --decay_type "linear" \
      --fp16 \
      --img_dir 'trainimage_path/training_images' \
      --classifier 'transfg' \
      --test_stage 2 \
      --img_labels 'train.txt'  \
      --img_val 'val.txt' 
      ```
4. Test 
   - Get my best trained model in this [link](https://reurl.cc/q1oZbN)
   - Run: 
      ```
      bash inference.sh
      ```
      by change some parameter (you can change the classifier by assign the 'classifier' : 'transfg' for TransFG and 'resnet50' for ResNet50)
      ```
      python -m torch.distributed.launch --nproc_per_node=1 inference.py \
      --img_order 'data_file/testing_img_order.txt' \
      --output 'data_file/answer.txt' \
      --annotation_file 'data_file/training_labels.txt' \
      --img_testdir '/path_to_testing_images' \
      --eval_batch_size 2 \
      --dataset 'myBirds' \
      --trained_model '/path/to_trainedmodel.bin' \
      --local_rank 0 \
      --classifier 'transfg' \
      --test_stage 2 \
      --pretrained_dir "vitpretrainedmodel_path/imagenet21k+imagenet2012_ViT-L_16.npz" \
      --model_type "ViT-L_16"
      ```
      
      
  data:
  1. Train cropped image download [here](https://drive.google.com/file/d/1qDok32E0L8zk1lNSqA29uK4AmOgYlXtE/view?usp=sharing)
  2. 
