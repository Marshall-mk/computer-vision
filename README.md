## History of computer vision architectures. A focus on Classification, Segmentation and Object detection networks.


| Paper | Date | Description |
|---|---|---|
| [Neocognition](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) | 1979 | A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position|
| [ConvNet](https://web.archive.org/web/20200110090230/http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) | 1989 | Used back-propagation to learn the convolution kernel coefficients directly from images of hand-written numbers |
| [Lenet](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4f26) | December 1998 | Introduced Convolutions. |
| [Alex Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f7c6) | September 2012 | Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012. |
| [ZfNet](https://arxiv.org/abs/1311.2901v3) | 2013 | ZFNet is a classic convolutional neural network. The design was motivated by visualizing intermediate feature layers and the operation of the classifier. Compared to AlexNet, the filter sizes are reduced and the stride of the convolutions are reduced. |
| [GoogleNet](https://arxiv.org/abs/1409.4842) | 2014 | One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection. |
| [VGG](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#c122) | September 2014 | Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014. |
| [Inception Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3) | September 2014 | Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales. |
| [HighwayNet](https://arxiv.org/abs/1505.00387) | 2015 | Introduced a new architecture designed to ease gradient-based training of very deep networks |
| [Inception Net v2 / Inception Net v3](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3) | December 2015 | Design Optimizations of the Inception Modules which improved performance and accuracy. |
| [Res Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f761) | December 2015 | Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015. |
| [Inception Net v4 / Inception ResNet](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#83ad) | February 2016 | Hybrid approach combining Inception Net and ResNet. |
| [Dense Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#65e8) | August 2016 | Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features. |
| [DarkNet](https://paperswithcode.com/method/darknet-53) | 2016 | A convolutional neural network that acts as a backbone for the YOLOv3 object detection approach. |
| [Xception](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#bc70) | October 2016 | Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules. |
| [Res Next](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#90bd) | November 2016 | Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups. |
| [FractalNet](https://arxiv.org/abs/1605.07648) | 2017 | The first simple alternative to ResNet. |
| [Capsule Networks](https://arxiv.org/abs/1710.09829) | 2017 | Proposed to improve the performance of CNNs, especially in terms of spatial hierarchies and rotation invariance. |
| [WideResNet](https://arxiv.org/abs/1605.07146) | 2017 | This paper first introduces a simple principle for reducing the descriptions of event sequences without loss of information. |
| [PolyNet](https://arxiv.org/abs/1611.05725) | 2017 | This paper proposes a novel synthetic network management model based on ForCES. This model regards the device under management (DUM) as forwarding element (FE). |
| [Pyramidal Net](https://arxiv.org/abs/1610.02915) | 2017 | A PyramidNet is a type of convolutional network where the key idea is to concentrate on the feature map dimension by increasing it gradually instead of by increasing it sharply at each residual unit with downsampling. In addition, the network architecture works as a mixture of both plain and residual networks by using zero-padded identity-mapping shortcut connections when increasing the feature map dimension. |
| [Squeeze and Excitation Nets](https://arxiv.org/abs/1709.01507) | 2017 | Focus on the channel relationship and propose a novel architectural unit, termed the "Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. These blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. |
| [Mobile Net V1](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#3cb5) | April 2017 | Uses depthwise separable convolutions to reduce the number of parameters and computation required. |
| [CMPE-SE](https://arxiv.org/pdf/1807.08920) | 2018 | Competitive squeeze and excitation networks |
| [RAN](https://arxiv.org/abs/1704.06904) | 2018 | Residual attention neural network. Residual Attention Network is built by stacking Attention Modules which generate attention-aware features. The attention-aware features from different modules change adaptively as layers going deeper.  |
| [CB-CNN](https://arxiv.org/abs/1804.08528) | 2018 | Channel boosted CNN, This idea of Channel Boosting exploits both the channel dimension of CNN (learning from multiple input channels) and Transfer learning (TL). TL is utilized at two different stages; channel generation and channel exploitation. |
| [CBAM](https://arxiv.org/abs/1807.06521) | 2018 |  Convolutional Block Attention Module, a simple yet effective attention module for feed-forward convolutional neural networks. Given an intermediate feature map, the module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement. |
| [Mobile Net V2](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4440) | January 2018 | Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks. |
| [Mobile Net V3](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#8eb6) | May 2019 | Uses AutoML to find the best possible neural network architecture for a given problem. |
| [Efficient Net](https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#560a) | May 2019 | Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost. |
| [Vision Transformer](https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b) | October 2020 | Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer |
| [SwAV](https://arxiv.org/abs/2006.09882) | 2020 | Self-supervised learning approach for image classification |
| [ResNesT](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.html) | 2022 | Designed to scale ResNet-style models to new levels of performance |
| [DeiT](https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec) | December 2020 | A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens. |
| [Swin Transformer](https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e) | March 2021 | A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision. |
| [CaiT](https://paperswithcode.com/method/cait) | 2021 | Combines vision transformers with convolutional layers |
| [T2T-ViT](https://arxiv.org/abs/2101.11986) | 2021 | Improved transformer-based vision models with token-to-token vision transformers. |
| [TNT](https://arxiv.org/abs/2103.00112) | 2021 | Transformer in Transformer architecture for better hierarchical feature learning |
| [BEiT](https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01) | June 2021 | Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers. |
| [MobileViT](https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434) | October 2021 | A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs. |
| [Masked AutoEncoder](https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af) | November 2021 | An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision. |
| [CoAtNet](https://arxiv.org/abs/2106.04803) | 2021 | CoAtNets (Convolution and Self-Attention Network)  |
| [ConvNeXt](https://arxiv.org/abs/2201.03545) | 2021 | A design that adopts a transformer-like architecture while being a convolutional network. It improves upon the designs of earlier CNNs.  |
| [NFNet](https://arxiv.org/pdf/2102.06171.pdf) | 2021 | High-Performance Large-Scale Image Recognition Without Normalization |
| [MLP-Mixer](https://arxiv.org/abs/2105.01601) | 2021 | Introduced mixer layers as an alternative to convolutional layers. |
| [gMLP](https://arxiv.org/abs/2111.03940) | 2021 | Gated activations for better gradient flow |
| [Conv Mixer](https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526) | January 2022 | Processes image patches using standard convolutions for mixing spatial and channel dimensions. |
| [MViT](https://arxiv.org/abs/2104.11227) | 2022 | A multiview vision transformer, designed for processing videos, providing a way to integrate information from different frames efficiently. |
| [Shuffle Transformer](https://arxiv.org/abs/2106.03650) | 2022 | Combined shuffle units with transformer blocks for efficient processing |
| [BEiT](https://arxiv.org/abs/2106.08254) | 2022 | Introduces a BERT-style pre-training approach for image recognition, using masked image modeling. |
| [CrossViT](https://arxiv.org/abs/2103.14899) | 2022 | Combines vision transformers with convolutional layers |
| [Masked Autoencoders (MAE)](https://arxiv.org/abs/2111.06377) | 2022 | A self-supervised learning method where the model learns to reconstruct images from partial inputs, improving efficiency and performance. |
| [RegNet](https://arxiv.org/pdf/2101.00590) | 2023 | Introduced a design space exploration approach to neural network architecture search, producing efficient and high-performing models for image classification and other tasks |

## Object Detection

| Paper | Date | Description |
|---|---|---|
| [RCNN](https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab) | November 2013 | Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression. |
| [SPPNet](https://arxiv.org/abs/1406.4729) | 2014 | Spatial Pyramid Pooling Network. |
| [Fast RCNN](https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0) | April 2015 | Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression. |
| [Faster RCNN](https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9) | June 2015 | A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features. |
| [YOLOv1](https://arxiv.org/abs/1506.02640) | 2015 | You only look Once V1. |
| [SSD](https://ritvik19.medium.com/papers-explained-31-single-shot-multibox-detector-14b0aa2f5a97) | December 2015 | Discretizes bounding box outputs over a span of various scales and aspect ratios per feature map. |
| [RFCN](https://arxiv.org/abs/1605.06409) | 2016 | Region-based Fully Convolutional Networks. |
| [YOLOv2](https://arxiv.org/abs/1612.08242) | 2016 | You only look Once V2. |
| [Feature Pyramid Network](https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8) | December 2016 | Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids. |
| [Mask RCNN](https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261) | March 2017 | Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch. |
| [Focal Loss](https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1) | August 2017 | Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples. |
| [RetinaNet](https://paperswithcode.com/method/retinanet) | 2017 | A one-stage object detection model that utilizes a focal loss function to address class imbalance during training. |
| [Cascade RCNN](https://arxiv.org/abs/1712.00726) | 2018 | A multi-stage object detection architecture, the Cascade R-CNN, consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. The detectors are trained stage by stage, leveraging the observation that the output of a detector is a good distribution for training the next higher quality detector. |
| [YOLOv3](https://arxiv.org/abs/1804.02767) | 2018 | You only look Once V3. |
| [EfficientDet](https://arxiv.org/abs/1911.09070) | 2019 | This paper aims to tackle this problem by systematically studying various design choices of detector architectures. |
| [CenterNet](https://arxiv.org/abs/1904.08189) | 2019 | This paper presents an efficient solution which explores the visual patterns within each cropped region with minimal costs.  |
| [DETR](https://arxiv.org/abs/2005.12872) | 2020 | Detection Transformer, End-to-End Object Detection with Transformers, A new method that views object detection as a direct set prediction problem. |
| [YOLOv4](https://arxiv.org/abs/2004.10934) | 2020 | You only look Once V4. |
| [YOLOv5](https://www.sciencedirect.com/topics/computer-science/yolov5) | 2020 | You only look Once V5. |
| [YOLOv6](https://arxiv.org/abs/2209.02976) | 2022 | You only look Once V6. |
| [YOLOv7](https://arxiv.org/abs/2207.02696) | 2022 | You only look Once V7. |
| [YOLOv8](https://arxiv.org/abs/2305.09972) | 2023 | You only look Once V8. |
| [YOLO-NAS](https://deci.ai/blog/yolo-nas-object-detection-foundation-model/) | 2023 | The new YOLO-NAS architecture sets a new frontier for object detection tasks, offering the best accuracy and latency tradeoff performance. |
| [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) | 2023 | A cutting-edge end-to-end object detector that provides real-time performance while maintaining high accuracy. It leverages the power of Vision Transformers (ViT) to efficiently process multiscale features by decoupling intra-scale interaction and cross-scale fusion. RT-DETR is highly adaptable, supporting flexible adjustment of inference speed using different decoder layers without retraining. The model excels on accelerated backends like CUDA with TensorRT, outperforming many other real-time object detectors. |
| [SAM](https://docs.ultralytics.com/models/sam/) | 2023 |The Segment Anything Model, or SAM, is a cutting-edge image segmentation model that allows for promptable segmentation, providing unparalleled versatility in image analysis tasks. SAM forms the heart of the Segment Anything initiative, a groundbreaking project that introduces a novel model, task, and dataset for image segmentation. |
| [Fast-SAM](https://docs.ultralytics.com/models/fast-sam/) | 2023 | FastSAM is designed to address the limitations of the Segment Anything Model (SAM), a heavy Transformer model with substantial computational resource requirements. The FastSAM decouples the segment anything task into two sequential stages: all-instance segmentation and prompt-guided selection. The first stage uses YOLOv8-seg to produce the segmentation masks of all instances in the image. In the second stage, it outputs the region-of-interest corresponding to the prompt. |
| [Mobile-SAM](https://docs.ultralytics.com/models/mobile-sam/) | 2023 | Mobile Segment Anything (MobileSAM). |
| [YOLOv9](https://docs.ultralytics.com/models/yolov9/) | 2024 | You only look Once V9. |
| [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) | 2024 | YOLO-World tackles the challenges faced by traditional Open-Vocabulary detection models, which often rely on cumbersome Transformer models requiring extensive computational resources. These models' dependence on pre-defined object categories also restricts their utility in dynamic scenarios. YOLO-World revitalizes the YOLOv8 framework with open-vocabulary detection capabilities, employing vision-language modeling and pre-training on expansive datasets to excel at identifying a broad array of objects in zero-shot scenarios with unmatched efficiency. |

