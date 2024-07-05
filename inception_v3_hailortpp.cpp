#include "inception_v3_hailortpp.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>

InceptionV3Params::InceptionV3Params(const std::string &labels_file, float confidence_threshold)
    : confidence_threshold(confidence_threshold)
{
    std::ifstream file(labels_file);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
}

InceptionV3Params *init_inception_v3(const std::string &labels_file, float confidence_threshold)
{
    return new InceptionV3Params(labels_file, confidence_threshold);
}

void free_resources(void *params_void_ptr)
{
    InceptionV3Params *params = reinterpret_cast<InceptionV3Params *>(params_void_ptr);
    delete params;
}

void preprocess_inception_v3(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }

    auto input_tensor = roi->get_tensor("inception-v3/input_layer1");
    
    // Note: Implement or use an existing image resizing and normalization function
    // This is a placeholder and needs to be implemented based on your specific requirements
    // resize_and_normalize(input_tensor, 299, 299);
}

void postprocess_inception_v3(HailoROIPtr roi, void *params_void_ptr)
{
    if (!roi->has_tensors())
    {
        return;
    }

    InceptionV3Params *params = reinterpret_cast<InceptionV3Params *>(params_void_ptr);
    
    auto output_tensor = roi->get_tensor("inception-v3/fc1");
    
    auto *output_data = output_tensor->data();
    
    int max_index = std::distance(output_data, std::max_element(output_data, output_data + 1000));
    
    float confidence = output_data[max_index] / 255.0f;  // Assuming UINT8 output
    
    if (confidence >= params->confidence_threshold)
    {
        auto classification = HailoClassification(
            params->labels[max_index],
            confidence
        );
        
        roi->add_object(classification);
    }
}