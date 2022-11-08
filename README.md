# General yolov5-seg inference acceleration program 

cuda10.2 cudnn8.2.4 Tensorrt8.0.1.6 Opencv4.5.4  

## How to Run 

### 1.Train yolov5s-seg in your own dataset
please follow official code (yolov5-6.2)  

python segment/train.py  --data coco128-seg.yaml --weights yolov5s-seg.pt --cfg yolov5s-seg.yaml  

### 2.Gen tensort engine 
#### 2.1 official code to tensorrt  
python export.py --data coco128-seg.yaml --weights yolov5s-seg.pt --cfg yolov5s-seg.yaml --include engine  

#### 2.2 ONNX To Tensorrt  

##### 2.2.1 Gen onnx  
You can directly export the engine model with the official code of yolov5/v6.2, but the exported model can only be used on your current computer. I suggest you export the onnx model first, and then use the code I provide. The advantage is that even if you change the computer configuration, you can generate the engine you need as long as there is an onnx  

python export.py --data coco128-seg.yaml --weights yolov5s-seg.pt --cfg yolov5s-seg.yaml --include onnx  

a file 'yolov5s-seg.onnx' will be generated.  

##### 2.2.1 Gen engine
cd download file path  
copy 'yolov5s-seg.onnx' to models/  

mkdir build  
cd build  
cmake ..  
make  
// file onnx2trt and trt_infer will be generated  
sudo ./onnx2trt ../models/yolov5s-seg.onnx ../models/yolov5s-seg.engine  
// a file 'yolov5s-seg.engine' will be generated.  

### 3.Use tensorrt engine

sudo ./trt_infer  ../models/yolov5s-seg.engine ../imagaes/street.jpg  

'''  
for (int i = 0; i < 10; i++) {//计算10次的推理速度
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, prob1, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}
'''   
The inference time is stable at 10ms  (gtx 1080ti)  


![image](https://github.com/fish-kong/Yolov5-instance-seg-tensorrt/blob/main/output.jpg)
