import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import setproctitle
import hailo
from hailo_common_funcs import get_numpy_from_buffer, disable_qos
from hailo_rpi_common import get_default_parser, QUEUE, get_caps_from_pad, GStreamerApp, app_callback_class, display_user_data_frame

import os
import hailo
import numpy as np
import cv2

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_bbox = None
        self.prev_frame = None 
        self.prev_pts =  None 
        self.last_label = None 
        self.prev_roi = None
        self.orb = cv2.ORB_create()

user_data = user_app_callback_class()



def get_frame_info(pad, info):
    buffer = info.get_buffer()
    format, width, height = get_caps_from_pad(pad) if buffer else (None, None, None)
    return buffer, format, width, height



def app_callback(pad, info, user_data):
    buffer, format, width, height = get_frame_info(pad, info)
    if not all([buffer, format, width, height]): 
        return Gst.PadProbeReturn.OK

    # current_frame = get_numpy_from_buffer(buffer, format, width, height)
    return Gst.PadProbeReturn.OK

class GStreamerInstanceSegmentationApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)

        self.batch_size = 1
        self.network_width = 299
        self.network_height = 299
        self.network_format = "RGB"
    
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov5seg_post.so')

        self.default_network_name = "inception_v3"
        self.hef_path = os.path.join(self.current_path, '../inception_v3.hef')

        self.app_callback = app_callback
        self.source_type = "rpi"
        self.processing_path = os.path.join(self.current_path, "../libinception_v3_inference.so")
        setproctitle.setproctitle(" detection and tracking app")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_element = f"libcamerasrc name=src_0 auto-focus-mode=AfModeManual ! "
        source_element += f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
        source_element += QUEUE("queue_src_scale")
        source_element += f"videoscale ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "

        source_element += QUEUE("queue_scale")
        source_element += f" videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += f" videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += "tee name=t ! "
        pipeline_string += QUEUE("bypass_queue", max_size_buffers=20) + "hmux.sink_0 "
        pipeline_string += "t. ! " + QUEUE("queue_hailonet")
        pipeline_string += "videoconvert n-threads=3 ! "
        
        # Defining own postprocessing path                                                                                    self.processing_path
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
        pipeline_string += f"hailofilter function-name={self.default_network_name} so-path={self.default_postprocess_so} qos=false ! "
        #
        
        pipeline_string += QUEUE("queue_hmuc") + " hmux.sink_1 "
        pipeline_string += "hmux. ! "
        
        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        
        return pipeline_string
    

if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    app = GStreamerInstanceSegmentationApp(args, user_data)
    app.run()