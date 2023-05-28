/*
** 使用.cu是希望使用CUDA的编译器NVCC, 会自动连接cuda库

** TensorRT runtime 推理过程

** 1, 创建一个runtime对象
** 2, 反序列化生成engine: runtime ---> engine
** 3, 创建一个执行上下文ExecutionContext: engine ---> context
** 4, 填充数据
** 5, 执行推理: context ---> enqueueV2
** 6, 释放资源: delete
*/

#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

#include <NvInfer.h>
#include "cuda_runtime.h"

class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 加载模型
std::vector<unsigned char> loadEngineModel(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open() && "load engine model failed!");

    // 移动到文件末尾
    file.seekg(0, std::ios::end);
    // 获取文件大小
    size_t size = file.tellg();

    std::vector<unsigned char> data(size);
    // 移动到文件开始
    file.seekg(0, std::ios::beg);
    // 读取文件内容到data中
    file.read((char *)data.data(), size);

    // 关闭文件
    file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    // ====== 1. 创建一个runtime对象 ======
    TRTLogger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);

    // ====== 2. 反序列化生成engine ======
    auto engineModel = loadEngineModel("./model/mlp.engine");
    // 调用runtime的反序列化方法, 生成engine, 参数分别是: 模型数据地址, 模型大小, pluginFactory
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr);

    if (!engine)
    {
        std::cout << "deserialize engine failed!" << std::endl;

        return -1;
    }

    // ====== 3. 创建一个执行上下文 ======
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // ====== 4. 填充数据 ======
    // 设置stream流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 数据流转: host -> device -> inference -> host
    // 输入数据
    float *host_input_data = new float[3]{2, 4, 8};
    // 输入数据大小
    int input_data_size = 3 * sizeof(float);
    // device 输入数据
    float *device_input_data = nullptr;

    // 输出数据
    float *host_output_data = new float[2]{0, 0};
    // 输出数据大小
    int output_data_size = 2 * sizeof(float);
    // device 输出数据
    float *device_output_data = nullptr;

    // 申请device内存
    cudaMalloc((void **)&device_input_data, input_data_size);
    cudaMalloc((void **)&device_output_data, output_data_size);

    // host ---> device
    // 参数: 目标地址, 源地址, 数据大小, 拷贝方向
    cudaMemcpyAsync(device_input_data, host_input_data, input_data_size, cudaMemcpyHostToDevice, stream);

    // bindings告诉Context输入输出数据的位置
    float *bindings[] = {device_input_data, device_output_data};

    // ====== 5. 执行推理 ======
    bool success = context->enqueueV2((void **)bindings, stream, nullptr);

    // 数据从device ---> host
    cudaMemcpyAsync(host_output_data, device_output_data, output_data_size, cudaMemcpyDeviceToHost, stream);
    // 等待流执行完毕
    cudaStreamSynchronize(stream);
    // 输出结果
    std::cout << "输出结果: " << host_output_data[0] << " " << host_output_data[1] << std::endl;

    // ====== 6. 释放资源 ======
    cudaStreamDestroy(stream);
    cudaFree(device_input_data);
    cudaFree(device_output_data);

    delete host_input_data;
    delete host_output_data;
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
