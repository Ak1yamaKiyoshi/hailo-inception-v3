#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include <vector>
#include <string>

__BEGIN_DECLS

class InceptionV3Params
{
public:
    std::vector<std::string> labels;
    float confidence_threshold;

    InceptionV3Params(const std::string &labels_file = "./imagenet_classes.txt",
                      float confidence_threshold = 0.5f);
};

InceptionV3Params *init_inception_v3(const std::string &labels_file, float confidence_threshold);
void free_resources(void *params_void_ptr);
void preprocess_inception_v3(HailoROIPtr roi);
void postprocess_inception_v3(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS