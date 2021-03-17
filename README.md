# UW_CSE455_final_project
CSE 455 final project, using fast R-CNN to train a instance seg and object detection model on NuImages and provide result in CRUW dataset format.


## Introduction
This project is aiming for provide ground truth for [CRUW dataset](https://www.cruwdataset.org/introduction) built by Information Processing Lab @UWECE.This devkit  provide image base Mask RCNN groud truth result as benchmark in cruw format. The object detection base on [Detectron2](https://github.com/facebookresearch/detectron2) is the main part for this project. And use transformed [nuImages](https://www.nuscenes.org/nuimages) dataset to pretrain the benchmark model. This is a futher improvement for the last [CSE599 G deep learning class final project](https://github.com/TedSongjh/CSE599-fianl-project). I added the cyclist detection part and improve instance segmantation accuracy by 24.13% and object detection average precision by 23.74%.

## Related Work

[Detectron2](https://github.com/facebookresearch/detectron2) 

Detectron 2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. It is a ground-up rewrite of the previous version, Detectron, and it originates from maskrcnn-benchmark. This object detection algorithms is powered by the PyTorch deep learning framework,and also includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend, DeepLab, etc. I use mask R-CNN part of this project, and modify the source code to custom dataset.

[nuImages](https://www.nuscenes.org/nuimages) and [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) 

Instead of using KITTI, nuImages is the latest dataset for autonomous driving. This dataset is extracted and re-organize from the orginal nuScense dataset. The annotating the 1.4 million images in nuScenes with 2d annotations would have led to highly redundant annotations, which are not ideal for training an object detector. So the nuImages dataset set out to label a more varied large-scale image dataset from nearly 500 logs (compared to 83 in nuScenes) of driving data. 
The seniero and quaility are similar to the CRUW we collected. And also, the metadata and annotation is extract from [nuScenes](https://www.nuscenes.org/nuscenes) dataset, which have more information in the relational database.

This devkit is used in this project to load nuImage dataset. And also, when we desgined our dataset format, we took some schema built idea from this dataset.


[CRUW dataset](https://www.cruwdataset.org/introduction)

CRUW is a camera-radar dataset for autonomous vehicle applications collected eariler this year by IPL@UWECE, we collect these data from E18 parking lot, road in side UW, city scenes and I-5 highway. It is a good resource for researchers to study FMCW radar data, that has high potential in the future autonomous driving. We will publish this dataset soon.Our dataset contains a pair of stereo cameras and 2 77GHz FMCW radar antenna array, both the camera and radar are calibrated and sychronized. During the last summer, I made the preprocess of this dataset, time-syschronize the camera and radar,and transfer the FMCW radar data to Range-Angle-Doppler(RAD) format

## Installation

To install Dectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

This devkit is used in this project to load nuImage dataset.
To use nuScenes devkit:
```
pip install nuscenes-devkit
```

Put [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/NuImages-RCNN-FPN.yaml) Config file in folder ```/detectron2/configs```

Put [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) dataset transformer in folder ```/detectron2/detectron2/data/datasets```

Put [nuimages_inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_inference.py), [visual&inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visual%26inference.py), [visualize_nuImages.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visualize_nuImages.py) in root folder of detectron2
## nuImages Data loader
**1. Convert nuImages dataset to CRUW dataset format**

The nuImages dataset have more attributes than our dataset, and also the nuImages's categories is in detail, which is meaning less for our Camera-Radar Fusion (CRF) annotation(The information extracted from radar can only provide accuracy location and velocity information, the feature of objects are compressed).This part of the code can be access from [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py).

Use the nuScence build in devkit to load nuImages dataset and convert the categories use a mapping function, read all the relational data and transfer the metadata as a dict. For the segmantation part, the orignal segmantation format is a single map with category IDs for each instance, convert the segmantation to each map per object, which can help with futher fusion in objects.And also, in nuImages, the cyclist are not seperated into different kind of pedestrain, but we want to merge the cyclist and the vehicle.cycle, so read the attribution annotation and the bicycle with rider will be train as different category.

The categories mapping from nuImages to CRUW is:
nuImages Category | CRUW Category
------------ | -------------
animal	|	-
human.pedestrian.adult	|	human.pedestrian
human.pedestrian.child	|	human.pedestrian
human.pedestrian.construction_worker	|	human.pedestrian
human.pedestrian.personal_mobility	|	human.pedestrian
human.pedestrian.police_officer	|	human.pedestrian
human.pedestrian.stroller	|	human.pedestrian
human.pedestrian.wheelchair	|	human.pedestrian
movable_object.barrier	|	-
movable_object.debris	|	-
movable_object.pushable_pullable	|	-
movable_object.trafficcone	|	-
static_object.bicycle_rack	|	-
vehicle.bicycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.bicycle(without attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.bus.bendy	|	vehicle.bus
vehicle.bus.rigid	|	vehicle.bus
vehicle.car	|	vehicle.car
vehicle.construction	|	vehicle.car
vehicle.emergency.ambulance	|	vehicle.car
vehicle.emergency.police	|	vehicle.car
vehicle.motorcycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.motorcycle(with attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.trailer	|	vehicle.truck
vehicle.truck	|	vehicle.truck
flat.drivable_surface	|	-
flat.ego	|	-

**2. Use Custom datasets on Detectron2**

After made the dataset reader, register the nuimages_test and nuimages_train dataset and metadata. use COCO InstanceSegmentation evaluator in the following part, and convert the nuImages format to CRUW dataset format, by changing object information schema, segmantation map to bitmask and bounding box format. This part is in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py). Change dataset and version name to register other dataset. 
```
dataset = 'nuimages_train'
version = 'v1.0-train'
root_path = '/mnt/disk1/nuImages_test/'
get_dicts = lambda p = root_path, c = categories: load_nuimages_dicts(path=p,version = version, categories=c)
DatasetCatalog.register(dataset,get_dicts)
MetadataCatalog.get(dataset).thing_classes = categories
MetadataCatalog.get(dataset).evaluator_type = "coco"
```


**3.Train nuImages use Mask R-CNN**

Train on nuImages v1.0-train dataset
First, change dataset and version in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) to
```
dataset = 'nuimages_train'
version = 'v1.0-train'
```


To train the dataset on Detectron2 useing ResNet FPN backbone. 

```
./detectron2/tools/train_net.py   --config-file ../configs/NuImages-RCNN-FPN.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

The detail of this archetecuture can be found in [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/NuImages-RCNN-FPN.yaml)

Train from last model (in this case is model_final.pth)
```
./detectron2/tools/train_net.py --num-gpus 1  --config-file ../configs/NuImages-RCNN-FPN.yaml MODEL.WEIGHTS ~/detectron2/tools/output-1/model_final.pth SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

**4.Evaluation on nuImages val dataset**

First, change dataset and version in [nuimages_loader.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_loader.py) to
```
dataset = 'nuimages_val'
version = 'v1.0-val'
```
Then run eval-only command and set model weights to the last checkpoint (in this case is model_final.pth)
```
./detectron2/tools/train_net.py    --config-file ../configs/NuImages-RCNN-FPN.yaml   --eval-only MODEL.WEIGHTS ~/detectron2/tools/output/model_final.pth
```

## Inference tools
There are three inference tools to visulize result

**1. Visulize nuImages groud truth**

run [visualize_nuImages.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visualize_nuImages.py)

**2. Inference on own dataset**

run [nuimages_inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/nuimages_inference.py) to save inference result in folder

**3. Inference on nuImages dataset and compare to ground truth**

run [visual&inference.py](https://github.com/TedSongjh/cruw_detectron2_devkit/blob/main/visual%26inference.py) to save both the groud truth and inference result

## Visual Results

Inference on NuImage val dataset:
![nuimage_val_1](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/n006-2018-09-17-11-57-46-0400__CAM_BACK__1537200799887532.png)

![nuimage_val_2](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/n008-2018-05-14-14-13-44-0400__CAM_BACK_LEFT__1526321708897295.png)

![nuimage_val_4](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/n008-2018-05-14-14-13-44-0400__CAM_FRONT__1526321757862465.png)

![nuimage_val_5](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/n010-2018-07-03-14-40-41%2B0800__CAM_BACK_LEFT__1530600537147633.png)

![nuimage_val_6](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/n010-2018-07-05-15-31-09%2B0800__CAM_BACK__1530776395387257.png)

Inference on CRUW dataset:

![curw_val_1](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000000000.jpg)


![curw_val_2](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000000049.jpg)


![curw_val_3](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000000385.jpg)

![curw_val_4](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000000419.jpg)

![curw_val_5](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000000644.jpg)

![curw_val_6](https://github.com/TedSongjh/UW_CSE455_final_project/blob/main/image/0000001022.jpg)
From above image, we can see that 

## Quantitative Results

Evaluation results for bbox: 

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 39.500 | 63.135 | 42.315 | 24.586 | 46.119 | 56.982 |


Evaluation results for segm: 

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 31.848 | 56.872 | 30.915 | 14.058 | 38.287 | 53.346 |




Per-category bbox AP: 

| category         | AP     | category      | AP     | category                 | AP     |
|:-----------------|:-------|:--------------|:-------|:-------------------------|:-------|
| human.pedestrian | 34.001 | vehicle.car   | 51.809 | vehicle.bus              | 33.752 |
| vehicle.truck    | 33.731 | vehicle.cycle | 37.375 | vehicle.cycle.with_rider | 46.333 |


Per-category segm AP: 

| category         | AP     | category      | AP     | category                 | AP     |
|:-----------------|:-------|:--------------|:-------|:-------------------------|:-------|
| human.pedestrian | 24.440 | vehicle.car   | 43.297 | vehicle.bus              | 30.795 |
| vehicle.truck    | 28.962 | vehicle.cycle | 22.666 | vehicle.cycle.with_rider | 40.930 |



Intersection over Union(IoU) Evaluation AP and AR:
|Evaluation|  IoU   | Range  | MaxDets|Result|
|:--------:|:------:|:------:|:------:|:------:|
|Average Precision  (AP) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.318
|Average Precision  (AP) | IoU=0.50      | area=   all | maxDets=100 | 0.569
|Average Precision  (AP) | IoU=0.75      | area=   all | maxDets=100 | 0.309
|Average Precision  (AP) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.141
|Average Precision  (AP) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.383
|Average Precision  (AP) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.533
|Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=  1 | 0.274
|Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets= 10 | 0.398
|Average Recall     (AR) | IoU=0.50:0.95 | area=   all | maxDets=100 | 0.402
|Average Recall     (AR) | IoU=0.50:0.95 | area= small | maxDets=100 | 0.222
|Average Recall     (AR) | IoU=0.50:0.95 | area=medium | maxDets=100 | 0.463
|Average Recall     (AR) | IoU=0.50:0.95 | area= large | maxDets=100 | 0.628






