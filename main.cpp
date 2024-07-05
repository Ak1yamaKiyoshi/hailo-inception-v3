#include "hailo/hailort.hpp"
#include "hailo_common.hpp"
#include "inception_v3_hailortpp.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <hef_path>" << std::endl;
        return 1;
    }

    std::string hef_path = argv[1];

    try {
        auto params = init_inception_v3("./imagenet_classes.txt", 0.5f);

        // Initialize Hailo device
        auto device = hailort::Device::create();
        
        // Create VStreams
        auto vstreams = hailort::VStreams::create(*device, hef_path);

        // Create input and output vstreams
        auto input_vstream = vstreams->input_vstreams()[0];
        auto output_vstream = vstreams->output_vstreams()[0];

        // Allocate buffers for input and output
        std::vector<uint8_t> input_data(input_vstream->get_frame_size());
        std::vector<uint8_t> output_data(output_vstream->get_frame_size());

        // TODO: Load your image data into input_data

        // Create HailoROIPtr
        auto roi = std::make_shared<HailoROI>(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f));

        // Add input tensor to ROI
        auto input_tensor = std::make_shared<HailoTensor>(input_vstream->get_info(), input_data.data());
        roi->add_tensor(input_tensor);

        // Preprocess
        preprocess_inception_v3(roi);

        // Run inference
        input_vstream->write(input_data.data());
        output_vstream->read(output_data.data());

        // Add output tensor to ROI
        auto output_tensor = std::make_shared<HailoTensor>(output_vstream->get_info(), output_data.data());
        roi->add_tensor(output_tensor);

        // Postprocess
        postprocess_inception_v3(roi, params);

        // Print results
        for (auto obj : roi->get_objects()) {
            if (obj->get_type() == HAILO_CLASSIFICATION) {
                auto classification = std::dynamic_pointer_cast<HailoClassification>(obj);
                std::cout << "Label: " << classification->get_label() 
                          << ", Confidence: " << classification->get_confidence() << std::endl;
            }
        }

        // Cleanup
        free_resources(params);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}