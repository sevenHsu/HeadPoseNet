# Description
   Estimate head pose with CNN,using backbone convolution network such 
   as ResNet / MobileNet / ShuffleNet.Also try regress attention 
   direction vector \[x,y,z] directly and regress the Expectation of 
   classify softmax result.What's more,build soft label for classify.
   
# Usage 

## Training
   ```shell
   python train.py --mode [reg,cls,sord] --net [resnet,shufflenet,mobilenet]
   --num_classes [33,66] --num_epochs --lr --lr_decay --unfreeze 
   --train_data --valid_data --input_size [224,196,160,128,96] 
   --width_mult [1.0,0.5] --batch_size --top_k --cls2reg --alpha
   --save_dir
   ```
   - mode: method for predict last result,choice from [reg,cls,sord],reg
   means regress directly,cls means regress the expectation of classify
   softmax results,sord means using soft label for cls
   - net: the alternative backbone network
   - num_classes: numbers of classify when mode is cls or sord
   - lr: learning rate
   - lr_decay: decay rate of learning rate,means lr=lr*lr_decay at 
   assigned time
   - unfreeze: for fine tune pretrained weight when epoch < unfreeze
   - train_data: path of training dataset
   - valid_data: path of validation dataset
   - input_size: size of input image for network
   - width_mult: value of width_mult for mobilenet
   - batch_size: training and validation batch size
   - top_k: top k of classify sofmax result
   - cls2reg: method for calculate regression value after classify
   - alpha: weight of regression loss
   - save_dir: path of saving model weight snapshot

## Testing
   ```shell
   python test.py --mode [reg,cls,sord] --net [resnet,shufflenet,mobilenet]
   --num_classes [33,66] --test_data --input_size [224,196,160,128,96] 
   --width_mult [1.0,0.5] --batch_size --top_k --cls2reg --alpha
   --snapshot --show_front --analysis --huge_error --collect_score
   --save_dir
   ```
   - mode: method for predict last result,choice from [reg,cls,sord],reg
   means regress directly,cls means regress the expectation of classify
   softmax results,sord means using soft label for cls
   - net: the alternative backbone network
   - num_classes: numbers of classify when mode is cls or sord
   - test_data: path of testing dataset
   - input_size: size of input image for network
   - width_mult: value of width_mult for MobileNet
   - batch_size: training and validation batch size
   - top_k: top k of classify softmax result
   - cls2reg: method for calculate regression value after classify
   - alpha: weight of regression loss
   - snapshot: path of model weight
   - show_front: draw gt axis and predicted vector in originaa images and save  
   - analysis: analysis degrees distribution on euler angles and save analysis results
   - huge_error: choice worst degrees predicted images and save
   - collect_score: dump degrees error on every image into a json file
   
## Experiment
训练数据集：

|         | Blur(P) | gary(P) | amount | type     |
|:--------|:--------|:--------|:-------|:---------|
| 300W_LP | 0.05    | 0.85    | 126k   | color    |
| DMS591  | 0.05    | 0       | 19K    | infrared |

模型对比：评价指标（degrees mae/collect score(limit=10º)）

|                 | pretrained | width_mult | input_size | Flops | Test on AFLW | Test on DMS |
|:----------------|:-----------|:-----------|:-----------|:------|:-------------|:------------|
| MobileNetV2     | Yes        | 1.0        | 224        | 299   | 5.9º/88%     | 5.5º/89%    |
| MobileNetV2     | Yes        | 1.0        | 128        | 97    | 6.3º/87%     | 5.3º/90%    |
| MobileNetV2     | No         | 0.5        | 224        | 87    | 7.7º/78%     | 6.3º/84%    |
| MobileNetV2     | No         | 0.5        | 128        | 28    | 9.3º/68%     | 7.7º/74%    |
| SqueezeNet1_1   | Yes        |            | 128        | 87    | 6.3º/85%     | 5.6º/89%    |
| shufflenet(1.0) | Yes        |            | 160        | 73    | 7.0º/81.62%  | 6.0º/85.9%  |

