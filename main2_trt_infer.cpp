#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>
using namespace nvinfer1;
using namespace cv;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int _segWidth = 160;
static const int _segHeight = 160;
static const int _segChannels = 32;
static const int CLASSES = 80;
static const int Num_box = 25200;
static const int OUTPUT_SIZE = Num_box * (CLASSES+5 + _segChannels);//output0
static const int OUTPUT_SIZE1 = _segChannels * _segWidth * _segHeight ;//output1


static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";//detect
const char* OUTPUT_BLOB_NAME1 = "output1";//mask


struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};

void DrawPred(Mat& img,std:: vector<OutputSeg> result) {
	//生成随机颜色
	std::vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < CLASSES; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);

		std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); //将mask加在原图上面


}



static Logger gLogger;
void doInference(IExecutionContext& context, float* input, float* output, float* output1, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
	const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
	// cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
	// cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输 
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	CHECK(cudaFree(buffers[outputIndex1]));
}



int main(int argc, char** argv)
{
	if (argc < 2){
		argv[1] = "../models/yolov5s-seg.engine";
		argv[2] = "../images/bus.jpg";
	}
    // create a model using the API directly and serialize it to a stream
	char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;  开辟空指针后 要和new配合使用，比如89行 trtModelStream = new char[size]
    size_t size{0};//与int固定四个字节不同有所不同,size_t的取值range是目标平台下最大可能的数组尺寸,一些平台下size_t的范围小于int的正数范围,又或者大于unsigned int. 使用Int既有可能浪费，又有可能范围不够大。
    
    std::ifstream file(argv[1], std::ios::binary);
    if (file.good()) {
		std::cout<<"load engine success"<<std::endl;
        file.seekg(0, file.end);//指向文件的最后地址
        size = file.tellg();//把文件长度告诉给size
		//std::cout << "\nfile:" << argv[1] << " size is";
		//std::cout << size << "";
		
		file.seekg(0, file.beg);//指回文件的开始地址
        trtModelStream = new char[size];//开辟一个char 长度是文件的长度
		assert(trtModelStream);//
        file.read(trtModelStream, size);//将文件内容传给trtModelStream
        file.close();//关闭
    }
	else {
		std::cout << "load engine failed" << std::endl;
		return 1;
	}
	
	Mat src=imread(argv[2]);
	if (src.empty()) {std::cout << "image load faild" << std::endl;return 1;}
	int img_width = src.cols;
	int img_height = src.rows;

    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
	Mat pr_img0, pr_img;
	std::vector<int> padsize;
	pr_img=preprocess_img(src, INPUT_H, INPUT_W,padsize);       // Resize
	//cv::resize(src, pr_img,Size(INPUT_H, INPUT_W), 0, 0, cv::INTER_LINEAR);
	
	int i = 0;// [1,3,INPUT_H,INPUT_W]
	//std::cout << "pr_img.step" << pr_img.step << std::endl;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 就是每一行有width个3通道的值
		for (int col = 0; col < INPUT_W; ++col)
		{
			
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
	
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
	static float prob[OUTPUT_SIZE];
	static float prob1[OUTPUT_SIZE1];
	
	//for (int i = 0; i < 10; i++) {//计算10次的推理速度
 //       auto start = std::chrono::system_clock::now();
 //       doInference(*context, data, prob, prob1, 1);
 //       auto end = std::chrono::system_clock::now();
 //       std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
 //   }

	doInference(*context, data, prob, prob1, 1);
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<std::vector<float>> picked_proposals;  //存储output0[:,:, 5 + _className.size():net_width]用以后续计算mask

	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	//printf("newh:%d,neww:%d,padh:%d,padw:%d", newh, neww, padh, padw);
	float ratio_h = (float)src.rows / newh;
	float ratio_w = (float)src.cols / neww;

	// 处理box
	int net_width = CLASSES + 5 + _segChannels;
	float* pdata = prob;
	for (int j = 0; j < Num_box; ++j) {
		float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
		if (box_score >= CONF_THRESHOLD) {
			cv::Mat scores(1, CLASSES, CV_32FC1, pdata + 5);
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= CONF_THRESHOLD) {

				std::vector<float> temp_proto(pdata + 5 + CLASSES, pdata + net_width);
				picked_proposals.push_back(temp_proto);

				float x = (pdata[0] - padw) * ratio_w;  //x
				float y = (pdata[1] - padh) * ratio_h;  //y
				float w = pdata[2] * ratio_w;  //w
				float h = pdata[3] * ratio_h;  //h

				int left = MAX((x - 0.5 * w) , 0);
				int top = MAX((y - 0.5 * h) , 0);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre * box_score);
				boxes.push_back(Rect(left, top, int(w ), int(h )));
			}
		}
		pdata += net_width;//下一行
	}
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
	std::vector<std::vector<float>> temp_mask_proposals;
	Rect holeImgRect(0, 0, src.cols, src.rows);
	std::vector<OutputSeg> output;
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		output.push_back(result);

		temp_mask_proposals.push_back(picked_proposals[idx]);

	}

	// 处理mask
	Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		//std::cout<< Mat(temp_mask_proposals[i]).t().size();
		maskProposals.push_back( Mat(temp_mask_proposals[i]).t() );

	pdata = prob1;
	std::vector<float> mask(pdata, pdata + _segChannels * _segWidth * _segHeight);
	
	Mat mask_protos = Mat(mask);
	Mat protos = mask_protos.reshape(0, { _segChannels,_segWidth * _segHeight });//将prob1的值 赋给mask_protos

	
	Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时
	Mat masks = matmulRes.reshape(output.size(), { _segWidth,_segHeight });
	//std::cout << protos.size();
	std::vector<Mat> maskChannels;
	split(masks, maskChannels);
	//std::cout << maskChannels.size();
	for (int i = 0; i < output.size(); ++i) {
		Mat dest, mask;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);//160*160
		
		Rect roi(int((float)padw / INPUT_W * _segWidth), int((float)padh / INPUT_H * _segHeight), int(_segWidth - padw / 2), int(_segHeight - padh / 2));
		//std::cout << roi;
		dest = dest(roi);
		
		resize(dest, mask, src.size(), INTER_NEAREST);


		//crop----截取box中的mask作为该box对应的mask
		Rect temp_rect = output[i].box;
		mask = mask(temp_rect) > MASK_THRESHOLD;

		//Point classIdPoint;
		//double max_class_socre;
		//minMaxLoc(mask, 0, &max_class_socre, 0, &classIdPoint);
		//max_class_socre = (float)max_class_socre;
		//printf("最大值:%.2f", max_class_socre);

		output[i].boxMask = mask;
	}



	DrawPred(src, output);

	cv::imwrite("output.jpg", src);
	char c = cv::waitKey(1);
    
	
	
	
	// Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

	system("pause");
    return 0;
}
