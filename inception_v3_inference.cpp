#include "hailo/hailort.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

constexpr int WIDTH = 299;
constexpr int HEIGHT = 299;

using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;

template <typename T, typename A>
int argmax(std::vector<T, A> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

std::vector<std::string> load_imagenet_classes(const std::string& file_path = "imagenet_classes.txt") {
    std::vector<std::string> classes;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Unable to open file " << file_path << std::endl;
        std::cerr << "Falling back to numbered classes." << std::endl;
    } else {
        std::string line;
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
            
            if (!line.empty()) {
                classes.push_back(line);
            }
        }
    }

    if (classes.empty()) {
        std::cerr << "Warning: No classes loaded from " << file_path << std::endl;
        std::cerr << "Using numbered classes instead." << std::endl;
        for (int i = 0; i < 1000; ++i) {  // Assuming 1000 classes for ImageNet
            classes.push_back("Class_" + std::to_string(i));
        }
    } else {
        std::cout << "Loaded " << classes.size() << " classes from " << file_path << std::endl;
    }

    return classes;
}


std::string getCmdOption(int argc, char *argv[], const std::string &option)
{
    std::string cmd;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0))
        {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = arg.substr(found, 200);
            return cmd;
        }
    }
    return cmd;
}


Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(Device &device, const std::string &hef_file)
{
    auto hef = Hef::create(hef_file);
    if (!hef) {
        return make_unexpected(hef.status());
    }

    auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = device.configure(hef.value(), configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}


template <typename T=InputVStream>
std::string info_to_str(T &stream)
{
    std::string result = stream.get_info().name;
    result += " (";
    result += std::to_string(stream.get_info().shape.height);
    result += ", ";
    result += std::to_string(stream.get_info().shape.width);
    result += ", ";
    result += std::to_string(stream.get_info().shape.features);
    result += ")";
    return result;
}


template <typename T>
hailo_status write_all(std::vector<InputVStream> &input, std::string &image_path)
{
    std::cout << "Write: Starting write process" << std::endl;
    if (input.empty() || image_path.empty()) {
        std::cerr << "Write: Invalid input parameters" << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }

    std::cout << "Write: Loading image from " << image_path << std::endl;
    auto rgb_frame = cv::imread(image_path, cv::IMREAD_COLOR);
    if (rgb_frame.empty()) {
        std::cerr << "Write: Failed to load image" << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }
    
    std::cout << "Write: Converting image to RGB" << std::endl;
    cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_BGR2RGB);

    std::cout << "Write: Resizing image to " << WIDTH << "x" << HEIGHT << std::endl;
    cv::resize(rgb_frame, rgb_frame, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_AREA);
    
    std::cout << "Write: Image size after resize: " << rgb_frame.cols << "x" << rgb_frame.rows << std::endl;
    std::cout << "Write: Image channels: " << rgb_frame.channels() << std::endl;

    std::cout << "Write: Convert to CV-32FC3 " << std::endl;;

    // Convert to float and normalize
    cv::Mat float_image;
    rgb_frame.convertTo(float_image, CV_32FC3, 1.0/255.0);

    std::cout <<"Write: input buffer prepared"<< std::endl;;
    // Prepare the input buffer
    size_t expected_size = 1072812;  // As per the error message
    std::vector<float> input_buffer(expected_size / sizeof(float));

    // Copy the image data to the input buffer
    std::cout <<  "Write: Copy to buffer " << std::endl;
    size_t image_size = WIDTH * HEIGHT * 3;
    std::memcpy(input_buffer.data(), float_image.data, image_size * sizeof(float));

    std::cout << "Write: Writing to input vstream, size: " << expected_size << std::endl;
    try {
        auto status = input[0].write(MemoryView(input_buffer.data(), expected_size));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Write: Failed to write to input vstream, status: " << status << std::endl;
            return status;
        }
    } catch (const std::exception& e) {
        std::cerr << "Write: Exception while writing to input vstream: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }

    std::cout << "Write: Write process completed successfully" << std::endl;
    return HAILO_SUCCESS;
}

template <typename T>
hailo_status read_all(OutputVStream &output, const std::vector<std::string>& classes, cv::Mat& frame)
{
    std::cout << "DEBUG: Entering read_all function" << std::endl;
    try {
        std::cout << "read_all: Getting frame size" << std::endl;
        size_t frame_size = output.get_frame_size();
        std::cout << "read_all: Frame size: " << frame_size << std::endl;

        std::cout << "read_all: Allocating data vector" << std::endl;
        std::vector<T> data(frame_size / sizeof(T));
        std::cout << "read_all: Data vector size: " << data.size() << std::endl;
        
        std::cout << "read_all: Reading from output vstream" << std::endl;
        auto status = output.read(MemoryView(data.data(), data.size() * sizeof(T)));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Read: Failed to read from output vstream" << std::endl;
            return status;
        }
        std::cout << "read_all: Successfully read from output vstream" << std::endl;

        std::cout << "read_all: Finding class with highest probability" << std::endl;
        auto max_it = std::max_element(data.begin(), data.end());
        size_t class_id = std::distance(data.begin(), max_it);
        float confidence = *max_it;
        std::cout << "read_all: Class ID: " << class_id << ", Confidence: " << confidence << std::endl;
        
        if (class_id < classes.size()) {
            std::cout << "read_all: Preparing text to be drawn" << std::endl;
            std::string label = classes[class_id] + " (" + std::to_string(confidence) + ")";
            
            std::cout << "read_all: Setting up text drawing parameters" << std::endl;
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.7;
            int thickness = 2;
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
            
            std::cout << "read_all: Calculating text position" << std::endl;
            cv::Point text_org(10, text_size.height + 10);

            std::cout << "read_all: Drawing rectangle" << std::endl;
            cv::rectangle(frame, text_org + cv::Point(0, baseline),
                          text_org + cv::Point(text_size.width, -text_size.height),
                          cv::Scalar(0, 0, 0), cv::FILLED);

            std::cout << "DEBread_allUG: Drawing text" << std::endl;
            cv::putText(frame, label, text_org, font_face, font_scale,
                        cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            
            std::cout << "Predicted class: " << classes[class_id] << " (ID: " << class_id << "), Confidence: " << confidence << std::endl;
        } else {
            std::cerr << "Read: Invalid class ID predicted: " << class_id << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

        std::cout << "read_all: Exiting read_all function successfully" << std::endl;
        return HAILO_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception in read_all function: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }
}



void print_net_banner(std::pair<std::vector<InputVStream>, std::vector<OutputVStream>> &vstreams) {
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    std::cout << "-I- Dir  Name                                     " << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.first)
        std::cout << "-I- IN:  " << info_to_str<InputVStream>(value) << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.second)
        std::cout << "-I- OUT: " << info_to_str<OutputVStream>(value) << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
}


template <typename IN_T, typename OUT_T>
hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs, std::string image_path, const std::vector<std::string>& classes)
{
    std::cout << "DEBUG: Entering infer function" << std::endl;

    // Check input parameters
    std::cout << "infer: Checking input parameters" << std::endl;
    std::cout << "infer: inputs size: " << inputs.size() << std::endl;
    std::cout << "infer: outputs size: " << outputs.size() << std::endl;
    std::cout << "infer: image_path: " << image_path << std::endl;
    std::cout << "infer: classes size: " << classes.size() << std::endl;

    if (inputs.empty() || outputs.empty() || image_path.empty() || classes.empty()) {
        std::cerr << "Infer: Invalid input parameters" << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }

    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;

    try {
        std::cout << "infer: Attempting to read image: " << image_path << std::endl;
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return HAILO_INVALID_ARGUMENT;
        }
        std::cout << "infer: Image read successfully. Size: " << frame.size() << std::endl;

        std::cout << "infer: Creating input thread" << std::endl;
        std::thread input_thread([&inputs, &image_path, &input_status]() { 
            try {
                std::cout << "infer: Starting write_all in input thread" << std::endl;
                input_status = write_all<IN_T>(inputs, image_path); 
                std::cout << "infer: write_all completed with status: " << input_status << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Exception in input thread: " << e.what() << std::endl;
                input_status = HAILO_INTERNAL_FAILURE;
            }
        });

        std::cout << "infer: Creating output thread" << std::endl;
        std::thread output_thread([&outputs, &classes, &output_status, &frame]() { 
            try {
                std::cout << "infer: Starting read_all in output thread" << std::endl;
                output_status = read_all<OUT_T>(outputs[0], classes, frame); 
                std::cout << "infer: read_all completed with status: " << output_status << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Exception in output thread: " << e.what() << std::endl;
                output_status = HAILO_INTERNAL_FAILURE;
            }
        });

        std::cout << "infer: Waiting for input thread to complete" << std::endl;
        input_thread.join();
        std::cout << "infer: Input thread joined" << std::endl;

        std::cout << "infer: Waiting for output thread to complete" << std::endl;
        output_thread.join();
        std::cout << "infer: Output thread joined" << std::endl;

        std::cout << "infer: Checking thread execution status" << std::endl;
        if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
            std::cerr << "Infer: Thread execution failed. Input status: " << input_status << ", Output status: " << output_status << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

        std::cout << "infer: Attempting to save processed image" << std::endl;
        std::string output_path = "processed_" + image_path;
        bool imwrite_result = cv::imwrite(output_path, frame);
        if (imwrite_result) {
            std::cout << "Processed image saved as: " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save processed image" << std::endl;
        }

        std::cout << "infer: Exiting infer function successfully" << std::endl;
        return HAILO_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception in infer function: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }
}


int main(int argc, char**argv)
{
    try {
        std::cout << "Step 1: Parsing command line arguments" << std::endl;
        std::string hef_file = getCmdOption(argc, argv, "-hef=");
        std::string image_path = getCmdOption(argc, argv, "-path=");
        
        if (hef_file.empty() || image_path.empty()) {
            std::cerr << "Error: HEF file or image path is empty" << std::endl;
            return HAILO_INVALID_ARGUMENT;
        }

        std::cout << "-I- image path: " << image_path << std::endl;
        std::cout << "-I- hef: " << hef_file << std::endl;

        std::cout << "Step 2: Scanning for PCIe devices" << std::endl;
        auto all_devices = Device::scan_pcie();
        if (all_devices->empty()) {
            std::cerr << "Error: No PCIe devices found" << std::endl;
            return HAILO_INVALID_OPERATION;
        }

        std::cout << "Step 3: Creating PCIe device" << std::endl;
        auto device = Device::create_pcie(all_devices.value()[0]);
        if (!device) {
            std::cerr << "Error: Failed to create PCIe device" << std::endl;
            return device.status();
        }

        std::cout << "Step 4: Configuring network group" << std::endl;
        auto network_group = configure_network_group(*device.value(), hef_file);
        if (!network_group) {
            std::cerr << "Error: Failed to configure network group" << std::endl;
            return network_group.status();
        }
        
        std::cout << "Step 5: Creating vstream params" << std::endl;
        auto input_vstream_params = network_group.value()->make_input_vstream_params(true, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        
        if (!input_vstream_params || !output_vstream_params) {
            std::cerr << "Error: Failed to create vstream params" << std::endl;
            return HAILO_INVALID_OPERATION;
        }

        std::cout << "Step 6: Creating input and output vstreams" << std::endl;
        auto input_vstreams = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
        auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
        if (!input_vstreams || !output_vstreams) {
            std::cerr << "Error: Failed to create vstreams" << std::endl;
            return input_vstreams.status();
        }
        auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

        std::cout << "Step 7: Printing network banner" << std::endl;
        print_net_banner(vstreams);

        std::cout << "Step 8: Activating network group" << std::endl;
        auto activated_network_group = network_group.value()->activate();
        if (!activated_network_group) {
            std::cerr << "Error: Failed to activate network group" << std::endl;
            return activated_network_group.status();
        }

        std::cout << "Step 9: Loading ImageNet classes" << std::endl;
        std::vector<std::string> classes = load_imagenet_classes("imagenet_classes.txt");
        if (classes.empty()) {
            std::cerr << "Error: Failed to load ImageNet classes" << std::endl;
            return HAILO_INVALID_OPERATION;
        }
        
        std::cout << "Step 10: Running inference" << std::endl;
        auto status = infer<float, float>(vstreams.first, vstreams.second, image_path, classes);
        //auto status = infer<float32_t, float32_t>(vstreams.first, vstreams.second, image_path, classes);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Error: Inference failed" << std::endl;
            return status;
        }

        std::cout << "Program completed successfully" << std::endl;
        return HAILO_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception in main function: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }
}