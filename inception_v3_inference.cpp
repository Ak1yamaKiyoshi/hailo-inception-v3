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

std::vector<std::string> load_imagenet_classes(const std::string& file_path) {
    std::vector<std::string> classes;
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('\t');
        if (pos != std::string::npos) {
            classes.push_back(line.substr(pos + 1));
        }
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
    auto rgb_frame = cv::imread(image_path, cv::IMREAD_COLOR);
    
    if (rgb_frame.channels() == 3)
        cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_BGR2RGB);

    if (rgb_frame.rows != HEIGHT || rgb_frame.cols != WIDTH)
        cv::resize(rgb_frame, rgb_frame, cv::Size(WIDTH, HEIGHT), cv::INTER_AREA);
    
    int factor = std::is_same<T, uint8_t>::value ? 1 : 4;  // In case we use float32_t, we have 4 bytes per component
    auto status = input[0].write(MemoryView(rgb_frame.data, HEIGHT * WIDTH * 3 * factor)); // Writing HEIGHT * WIDTH, 3 channels of uint8
    if (HAILO_SUCCESS != status) 
        return status;

    return HAILO_SUCCESS;
}
template <typename T>
hailo_status read_all(OutputVStream &output, const std::vector<std::string>& classes)
{
    try {
        std::vector<T> data(output.get_frame_size() / sizeof(T));
        
        auto status = output.read(MemoryView(data.data(), data.size() * sizeof(T)));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to read from output vstream" << std::endl;
            return status;
        }

        // Here's where the prediction is made
        int class_id = argmax(data);
        
        if (class_id >= 0 && class_id < classes.size()) {
            std::cout << "Predicted class: " << classes[class_id] << " (ID: " << class_id << ")" << std::endl;
            
            // Optional: Print top 5 predictions
            std::vector<std::pair<float, int>> top_predictions;
            for (int i = 0; i < data.size(); ++i) {
                top_predictions.push_back({data[i], i});
            }
            std::partial_sort(top_predictions.begin(), top_predictions.begin() + 5, top_predictions.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            
            std::cout << "Top 5 predictions:" << std::endl;
            for (int i = 0; i < 5 && i < top_predictions.size(); ++i) {
                int id = top_predictions[i].second;
                float prob = top_predictions[i].first;
                std::cout << classes[id] << " (ID: " << id << "): " << prob << std::endl;
            }
        } else {
            std::cerr << "Invalid class ID predicted: " << class_id << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

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
    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;

    try {
        assert(!inputs.empty() && "Input vstreams vector is empty");
        assert(!outputs.empty() && "Output vstreams vector is empty");
        assert(!image_path.empty() && "Image path is empty");
        assert(!classes.empty() && "Classes vector is empty");

        std::thread input_thread([&inputs, &image_path, &input_status]() { 
            try {
                input_status = write_all<IN_T>(inputs, image_path); 
            } catch (const std::exception& e) {
                std::cerr << "Exception in input thread: " << e.what() << std::endl;
                input_status = HAILO_INTERNAL_FAILURE;
            }
        });

        std::thread output_thread([&outputs, &classes, &output_status]() { 
            try {
                output_status = read_all<OUT_T>(outputs[0], classes); 
            } catch (const std::exception& e) {
                std::cerr << "Exception in output thread: " << e.what() << std::endl;
                output_status = HAILO_INTERNAL_FAILURE;
            }
        });

        input_thread.join();
        output_thread.join();

        if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
            std::cerr << "Input status: " << input_status << ", Output status: " << output_status << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }

        std::cout << "-I- Inference finished successfully" << std::endl;
        return HAILO_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception in infer function: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }
}




int main(int argc, char**argv)
{
    try {
        std::string hef_file = getCmdOption(argc, argv, "-hef=");
        std::string image_path = getCmdOption(argc, argv, "-path=");
        
        assert(!hef_file.empty() && "HEF file path is empty");
        assert(!image_path.empty() && "Image path is empty");

        std::cout << "-I- image path: " << image_path << std::endl;
        std::cout << "-I- hef: " << hef_file << std::endl;

        auto all_devices = Device::scan_pcie();
        assert(!all_devices->empty() && "No PCIe devices found");

        auto device = Device::create_pcie(all_devices.value()[0]);
        if (!device) {
            std::cerr << "-E- Failed create_pcie " << device.status() << std::endl;
            return device.status();
        }

        auto network_group = configure_network_group(*device.value(), hef_file);
        if (!network_group) {
            std::cerr << "-E- Failed to configure network group " << hef_file << std::endl;
            return network_group.status();
        }
        
        auto input_vstream_params = network_group.value()->make_input_vstream_params(true, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        
        assert(input_vstream_params && "Failed to create input vstream params");
        assert(output_vstream_params && "Failed to create output vstream params");

        auto input_vstreams = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
        auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
        if (!input_vstreams or !output_vstreams) {
            std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
            return input_vstreams.status();
        }
        auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

        print_net_banner(vstreams);

        auto activated_network_group = network_group.value()->activate();
        if (!activated_network_group) {
            std::cerr << "-E- Failed activated network group " << activated_network_group.status();
            return activated_network_group.status();
        }

        std::vector<std::string> classes = load_imagenet_classes("words.txt");
        assert(!classes.empty() && "Failed to load ImageNet classes");
        
        auto status = infer<float32_t, float32_t>(vstreams.first, vstreams.second, image_path, classes);

        if (HAILO_SUCCESS != status) {
            std::cerr << "-E- Inference failed " << status << std::endl;
            return status;
        }

        return HAILO_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Exception in main function: " << e.what() << std::endl;
        return HAILO_INTERNAL_FAILURE;
    }
}