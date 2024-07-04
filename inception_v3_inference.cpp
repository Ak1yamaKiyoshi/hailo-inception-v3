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
            // Trim whitespace from the beginning and end of the line
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
    if (rgb_frame.channels() == 3)
        cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_BGR2RGB);

    std::cout << "Write: Resizing image to " << WIDTH << "x" << HEIGHT << std::endl;
    if (rgb_frame.rows != HEIGHT || rgb_frame.cols != WIDTH)
        cv::resize(rgb_frame, rgb_frame, cv::Size(WIDTH, HEIGHT), cv::INTER_AREA);
    
    int factor = std::is_same<T, uint8_t>::value ? 1 : 4;
    size_t expected_size = HEIGHT * WIDTH * 3 * factor;
    
    std::cout << "Write: Writing to input vstream, size: " << expected_size << std::endl;
    auto status = input[0].write(MemoryView(rgb_frame.data, expected_size));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Write: Failed to write to input vstream" << std::endl;
        return status;
    }

    std::cout << "Write: Write process completed successfully" << std::endl;
    return HAILO_SUCCESS;
}

template <typename T>
hailo_status read_all(OutputVStream &output, const std::vector<std::string>& classes)
{
    std::cout << "Read: Starting read process" << std::endl;
    
    try {
        size_t frame_size = output.get_frame_size();
        std::cout << "Read: Frame size: " << frame_size << std::endl;
        
        std::vector<T> data(frame_size / sizeof(T));
        std::cout << "Read: Allocated buffer of size: " << data.size() << std::endl;
        
        std::cout << "Read: Reading from output vstream" << std::endl;
        auto status = output.read(MemoryView(data.data(), data.size() * sizeof(T)));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Read: Failed to read from output vstream" << std::endl;
            return status;
        }

        std::cout << "Read: Processing output data" << std::endl;
        size_t class_id = static_cast<size_t>(argmax(data));
        
        if (class_id < classes.size()) {
            std::cout << "Predicted class: " << classes[class_id] << " (ID: " << class_id << ")" << std::endl;
            
            // Print top 5 predictions
            std::vector<std::pair<float, size_t>> top_predictions;
            for (size_t i = 0; i < data.size(); ++i) {
                top_predictions.push_back({data[i], i});
            }
            std::partial_sort(top_predictions.begin(), top_predictions.begin() + std::min(5ul, top_predictions.size()), top_predictions.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            
            std::cout << "Top 5 predictions:" << std::endl;
            for (size_t i = 0; i < std::min(5ul, top_predictions.size()); ++i) {
                size_t id = top_predictions[i].second;
                float prob = top_predictions[i].first;
                std::cout << classes[id] << " (ID: " << id << "): " << prob << std::endl;
            }
        } else {
            std::cerr << "Read: Invalid class ID predicted: " << class_id << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

        std::cout << "Read: Read process completed successfully" << std::endl;
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
    std::cout << "Infer: Starting inference process" << std::endl;
    
    if (inputs.empty() || outputs.empty() || image_path.empty() || classes.empty()) {
        std::cerr << "Infer: Invalid input parameters" << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }

    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;

    try {
        std::cout << "Infer: Starting input thread" << std::endl;
        std::thread input_thread([&inputs, &image_path, &input_status]() { 
            try {
                input_status = write_all<IN_T>(inputs, image_path); 
            } catch (const std::exception& e) {
                std::cerr << "Exception in input thread: " << e.what() << std::endl;
                input_status = HAILO_INTERNAL_FAILURE;
            }
        });

        std::cout << "Infer: Starting output thread" << std::endl;
        std::thread output_thread([&outputs, &classes, &output_status]() { 
            try {
                output_status = read_all<OUT_T>(outputs[0], classes); 
            } catch (const std::exception& e) {
                std::cerr << "Exception in output thread: " << e.what() << std::endl;
                output_status = HAILO_INTERNAL_FAILURE;
            }
        });

        std::cout << "Infer: Waiting for threads to complete" << std::endl;
        input_thread.join();
        output_thread.join();

        if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
            std::cerr << "Infer: Thread execution failed. Input status: " << input_status << ", Output status: " << output_status << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

        std::cout << "Infer: Inference finished successfully" << std::endl;
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
        auto status = infer<float32_t, float32_t>(vstreams.first, vstreams.second, image_path, classes);
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