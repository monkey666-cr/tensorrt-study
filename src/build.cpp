/*
** TensorRT build engine的创建过程
** 1, 创建builder
** 2, 创建网络定义: builder ---> network
** 3, 配置参数: builder ---> config
** 4, 生成engine: builder ---> engine (network, config)
** 5, 序列化保存: engine ---> serialize
** 6, 释放资源: delete
*/

#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <NvInfer.h>

// logger 用来管控打印日志级别
// TRTLogger 继承自 nvinfer1::ILogger
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity serverity, const char *msg) noexcept override
    {
        // 屏蔽INFO级别日志
        if (serverity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 保存权重
void saveWeights(const std::string &filename, const float *data, int size)
{
    std::ofstream outfile(filename, std::ios::binary);

    assert(outfile.is_open() && "save weights failed");

    // 保存权重大小
    outfile.write((char *)(&size), sizeof(int));
    // 保存权重数据
    outfile.write((char *)(data), size * sizeof(float));

    outfile.close();
}

// 读取权重
std::vector<float> loadWeights(const std::string &filename)
{
    std::ifstream input_file(filename, std::ios::binary);

    assert(input_file.is_open() && "load weight failed");

    int size;
    input_file.read((char *)(&size), sizeof(int));

    // 读取权重
    std::vector<float> data(size);
    input_file.read((char *)(data.data()), size * sizeof(float));

    input_file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    /* code */
    // ====== 1. 创建 builder ======
    TRTLogger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

    // ======= 2. 创建网络定义: builde3r ---> network ======

    // 显性 batch
    // 1 << 0 = 1, 二进制移位, 左移0位, 相当于1
    auto explicitBatch = 1u << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    // 定义网络结构
    // mlp 多层感知机: input(1, 3, 1, 1) ---> fc1 ---> sigmoid ---> output

    // 创建一个 input tensor, 参数分别是: name, data, type, dims
    const int input_size = 3;
    nvinfer1::ITensor *input = network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, input_size, 1, 1});

    // 创建全连接层fc1
    // weight and bias
    const float *fc1_weight_data = new float[input_size * 2]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    const float *fc1_bias_data = new float[2]{0.1, 0.5};

    // 将权重保存到文件中, 演示从别的来源加载权重
    // 保存权重和偏置
    saveWeights("model/fc1.wts", fc1_weight_data, 6);
    saveWeights("model/fc1.bias", fc1_bias_data, 2);

    // 加载权重和偏置
    auto fc1_weight_vec = loadWeights("model/fc1.wts");
    auto fc1_bias_vec = loadWeights("model/fc1.bias");

    // 转为 nvinfer1::Weights类型, 参数分别是: data type, data, size
    nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_vec.data(), fc1_weight_vec.size()};
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_vec.data(), fc1_bias_vec.size()};

    const int output_size = 2;
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size, fc1_weight, fc1_bias);

    // 添加激活层
    nvinfer1::IActivationLayer *sigmod = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    // 设置输出名字
    sigmod->getOutput(0)->setName("output");
    // 标记输出, 如果不标记会被当成瞬时值被优化掉
    network->markOutput(*sigmod->getOutput(0));

    // 设定最大的 batch size
    builder->setMaxBatchSize(1);

    // ====== 3. 配置参数: builder ---> config ======
    // 添加配置参数, 告诉TensorRT应该如何优化网络
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 设置最大工作空间
    config->setMaxWorkspaceSize(1 << 20); // 256MB

    // ====== 4. 创建 engine: builder ---> network ---> config ======
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::cerr << "Failed to create engine!" << std::endl;
        return -1;
    }

    // ====== 5. 序列化 engine ======
    nvinfer1::IHostMemory *serialized_engine = engine->serialize();
    // 存入文件
    std::ofstream outfile("model/mlp.engine", std::ios::binary);
    assert(outfile.is_open() && "Failed to open file for writing");
    outfile.write((char *)serialized_engine->data(), serialized_engine->size());

    // ====== 6. 释放资源 ======
    outfile.close();

    delete serialized_engine;
    delete engine;
    delete config;
    delete network;
    delete builder;

    std::cout << "engine文件生成成功" << std::endl;

    return 0;
}
