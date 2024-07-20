
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <MNN/Interpreter.hpp>
#include <opencv2/opencv.hpp>
class ImageDetector{
    public:
        std::shared_ptr<MNN::Interpreter> net_ = nullptr;
        void InitDetector();
        void PreProcess(const cv::Mat &input);
        MNN::Session *session_ = nullptr;
        int input_cols_=0;
        int input_rows_=0;        
    private:
        int model_height_ = 480;
        int model_width_ = 480;
        MNN::Tensor *input_tensor_;
        MNN::Tensor *head_P3_tensor_;
        MNN::Tensor *head_P4_tensor_;
        MNN::Tensor *head_P5_tensor_;
        MNN::Tensor * nhwc_Tensor_= nullptr;
};

