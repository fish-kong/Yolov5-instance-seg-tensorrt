#include <iostream>
#include "logging.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <fstream>

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;
int main(int argc,char** argv) {
	if (argc < 2){
		argv[1] = "../models/yolov5s-seg.onnx";
		argv[2] = "../models/yolov5s-seg.engine";
	}
	// 1 ����onnxģ��
	IBuilder* builder = createInferBuilder(gLogger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	const char* onnx_filename = argv[1];
	parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	std::cout << "successfully load the onnx model" << std::endl;

	// 2��build the engine
	unsigned int maxBatchSize = 1;
	builder->setMaxBatchSize(maxBatchSize);
	IBuilderConfig* config = builder->createBuilderConfig();
	//config->setMaxWorkspaceSize(1 << 20);
	config->setMaxWorkspaceSize(128 * (1 << 20));  // 16MB
	config->setFlag(BuilderFlag::kFP16);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	// 3��serialize Model
	IHostMemory *gieModelStream = engine->serialize();
	std::ofstream p(argv[2], std::ios::binary);
	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(gieModelStream->data()), gieModelStream->size());
	gieModelStream->destroy();


	std::cout << "successfully generate the trt engine model" << std::endl;
	return 0;
}