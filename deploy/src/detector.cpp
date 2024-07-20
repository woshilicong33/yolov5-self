#include <detector.hpp>


void ImageDetector::InitDetector(){
    std::string model_path = "./best_epoch_weights.mnn";
    auto handle       = dlopen("libMNN_CL.so", RTLD_NOW);
    net_              = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig netConfig;
    netConfig.type    = MNN_FORWARD_OPENCL;
    netConfig.numThread = 1;
    MNN::BackendConfig bConfig;
    bConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(MNN::BackendConfig::Precision_Low);
    netConfig.backendConfig = &bConfig;
    session_ = net_->createSession(netConfig);
    input_tensor_   = net_->getSessionInput(session_, "input_1");
    head_P3_tensor_ = net_->getSessionOutput(session_, "yolo_head_P3");
    head_P4_tensor_ = net_->getSessionOutput(session_, "yolo_head_P4");
    head_P5_tensor_ = net_->getSessionOutput(session_, "yolo_head_P5");


    std::vector<int> dims{1, (int)model_height_, (int)model_width_, 3};
    nhwc_Tensor_    = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    std::cout<<"Finish configuration"<<std::endl;
}


void ImageDetector::PreProcess(const cv::Mat &input) {
  input_cols_ = input.cols;
  input_rows_ = input.rows;
  cv::Mat raw_image = input.clone();
  cv::Mat image = cv::Mat::zeros(model_height_, model_width_, CV_32FC3);;
  cv::resize(raw_image, image, cv::Size(model_height_, model_width_));
  image.convertTo(image, CV_32FC3);
  image = image / 255.0f;
  auto nhwc_data = nhwc_Tensor_->host<float>();
  auto nhwc_size = nhwc_Tensor_->size();
  std::memcpy(nhwc_data, image.data, nhwc_size);
  input_tensor_->copyFromHostTensor(nhwc_Tensor_);
  for(int a=0;a<=10;a++){
    auto start_time = std::chrono::high_resolution_clock::now();
    net_->runSession(session_);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout<<"runSession:"<<duration.count()<<std::endl;

    auto head_P3_host_start_time = std::chrono::high_resolution_clock::now();   
    auto head_P3_host_ = std::make_shared<MNN::Tensor>(head_P3_tensor_, MNN::Tensor::CAFFE);
    head_P3_tensor_->copyToHostTensor(head_P3_host_.get());
    auto head_P3_host_end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(head_P3_host_end_time - head_P3_host_start_time);
    std::cout<<"head_P3:"<<duration.count()<<std::endl;

    auto head_P4_host_start_time = std::chrono::high_resolution_clock::now();   
    auto head_P4_host_ = std::make_shared<MNN::Tensor>(head_P4_tensor_, MNN::Tensor::CAFFE);
    head_P4_tensor_->copyToHostTensor(head_P4_host_.get());
    auto head_P4_host_end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(head_P4_host_end_time - head_P4_host_start_time);
    std::cout<<"head_P4:"<<duration.count()<<std::endl;

    auto head_P5_host_start_time = std::chrono::high_resolution_clock::now();   
    auto head_P5_host_ = std::make_shared<MNN::Tensor>(head_P5_tensor_, MNN::Tensor::CAFFE);
    head_P5_tensor_->copyToHostTensor(head_P4_host_.get());
    auto head_P5_host_end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(head_P5_host_end_time - head_P5_host_start_time);
    std::cout<<"head_P5:"<<duration.count()<<std::endl;

  }

}
