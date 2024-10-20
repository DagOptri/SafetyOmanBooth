#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds

perf_data = None

MAX_DISPLAY_LEN=64
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1080
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
pgie_classes_str= ["Person", "Bag", "Face"]
pgie_config_file = "pgie_peoplenet_config.txt"
tracker_config_file = "config_tracker_mot.txt"
sgie_config_file = "sgie_yolo_detector_config.txt"
analytics_config_file = "config_nvdsanalytics.txt"

def is_helmet_on_head(person_center_x, person_center_y, helmet_center_x, helmet_center_y, threshold=30):
    """Check if the helmet is positioned near the center of the person's head."""
    # Check if the helmet center is within a certain distance of the head center
    distance = ((person_center_x - helmet_center_x) ** 2 + (person_center_y - helmet_center_y) ** 2) ** 0.5
    return distance < threshold  # Return True if the helmet is roughly on the head

# nvanlytics_src_pad_buffer_probe  will extract metadata received on nvtiler sink pad
# and update params for drawing rectangle, object information etc.
def nvanalytics_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    # Set of persons IDs who has assosiated with helmets
    assosiated_persons_ids = set()
    skip_counter = 5

    while l_frame:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        obj_counter = {'person': 0, 'helmet': 0}
        helmet_locations = {}  # Store helmet detections with object ID and center positions
        person_info = []  # Store information about detected persons in restricted areas

        # print("#"*50)

        while l_obj:
            try: 
                # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                # The casting is done by pyds.NvDsObjectMeta.cast()
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.obj_label == "Helmet":
                # Calculate center of the helmet bounding box
                helmet_center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                helmet_center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                helmet_locations[obj_meta.object_id] = (helmet_center_x, helmet_center_y)
                obj_counter['helmet'] += 1

            if obj_meta.obj_label == "person":
                obj_counter['person'] += 1
                rect_params = obj_meta.rect_params
                in_restricted_area = False
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                        if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):             
                            user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                            #if user_meta_data.dirStatus: print("Object {0} moving in direction: {1}".format(obj_meta.object_id, user_meta_data.dirStatus))                    
                            #if user_meta_data.lcStatus: print("Object {0} line crossing status: {1}".format(obj_meta.object_id, user_meta_data.lcStatus))
                            #if user_meta_data.ocStatus: print("Object {0} overcrowding status: {1}".format(obj_meta.object_id, user_meta_data.ocStatus))
                            if user_meta_data.roiStatus and "Restricted Area" in user_meta_data.roiStatus:
                                in_restricted_area = True
                    except StopIteration:
                        break

                    try:
                        l_user_meta = l_user_meta.next
                    except StopIteration:
                        break
            
                # Store information for persons in restricted zones
                if in_restricted_area:
                    person_info.append((obj_meta.object_id, rect_params, obj_meta.text_params))

            rect_params = obj_meta.rect_params
            rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green border

            text_params = obj_meta.text_params
            # Modify the display text to show the class label and confidence
            text_params.display_text = "{0}".format(obj_meta.obj_label)

            # Optionally, change the font size and color
            text_params.font_params.font_size = 12  # Set font size
            text_params.font_params.font_name = "Serif"  # Change font style if needed

            # Set text color (RGBA format: 1.0 = full intensity)
            text_params.set_bg_clr = 1  # Enable background color

            text_params.text_bg_clr.set(0.0, 1.0, 0.0, 0.3) # Green background with 30% opacity

            # Set the text color (white, full intensity)
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        security_alert_cnt = len(person_info)
        # After gathering all detections, associate helmets with persons in the restricted area
        for person_id, rect_params, text_params in person_info:
            if person_id in assosiated_persons_ids:
                continue
            assosiated_persons_ids.add(person_id)
            helmet_associated = False
            treshold = rect_params.height / 12
            person_head_center_x = rect_params.left + rect_params.width / 2
            person_head_center_y = rect_params.top + rect_params.height / 12  # Approximate head location in upper 1/6th

            for _, (helmet_center_x, helmet_center_y) in helmet_locations.items():
                if is_helmet_on_head(person_head_center_x, person_head_center_y, helmet_center_x, helmet_center_y, treshold):
                    helmet_associated = True
                    break

            # Set bounding box color based on helmet association
            if helmet_associated:
                security_alert_cnt -= 1
                rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green border for wearing helmet
            else:
                rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Red border for not wearing helmet
                text_params.text_bg_clr.set(1.0, 0.0, 0.0, 0.3)  # Red background with 30% opacity
                text_params.display_text = pyds.get_string(text_params.display_text) + " DANGER ZONE: WEAR A HELMET!"
        
        display_text = "People Current Frame: {0}\n".format(obj_counter['person'])
        
        # Get meta data from NvDsAnalyticsFrameMeta
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
                    if (zone_dict := user_meta_data.objInROIcnt):
                        for zone, cnt in zone_dict.items():
                            display_text += "People in {0}: {1}\n".format(zone, cnt)
                    if (line_dict := user_meta_data.objLCCumCnt):
                        for line, cnt in line_dict.items():
                            display_text += "{0} Total: {1}\n".format(line, cnt)
                    if (oc_dict := user_meta_data.ocStatus):
                        for zone, status in oc_dict.items():
                            display_text += "Overcrowding in {0}: {1}\n".format(zone, "Yes" if status else "No")
                    display_text += "Security alerts: {0}".format(security_alert_cnt)
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break

        # Update skip counter
        skip_counter -= 1
        if skip_counter == 0:
            assosiated_persons_ids.clear()
            skip_counter = 5

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = display_text

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 18
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # print("Frame Number=", frame_number, "stream id=", frame_meta.pad_index, "Number of Objects=",num_rects,"Face_count=",obj_counter[PGIE_CLASS_ID_FACE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        #print("#"*50)

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    global perf_data
    perf_data = PERF_DATA(len(args) - 1)
    number_sources=len(args)-1

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.request_pad_simple(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvtracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    print("Creating nvinfer (SGIE1) \n ")
    
    # Create nvinfer element
    sgie = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to make sgie \n")

    print("Creating nvdsanalytics \n ")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write(" Unable to create nvanalytics \n")
    nvanalytics.set_property("config-file", analytics_config_file)

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        sink.set_property('sync', 0)
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', pgie_config_file)
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read(tracker_config_file)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)

    # Set config file for each SGIE
    sgie.set_property('config-file-path', sgie_config_file)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie)
    pipeline.add(nvanalytics)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # We link elements in the following order:
    # sourcebin -> streammux -> nvinfer -> nvtracker -> nvdsanalytics ->
    # nvtiler -> nvvideoconvert -> nvdsosd -> sink
    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(sgie)
    sgie.link(queue4)
    queue4.link(nvanalytics)
    nvanalytics.link(queue5)
    queue5.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    nvosd.link(queue8)
    queue8.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    nvanalytics_src_pad=nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
