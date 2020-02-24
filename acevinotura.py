import argparse
import json

import cv2
import os
import sys
from tqdm import tqdm
from openvino.inference_engine import IENetwork, IECore, np

MODEL = "models/frozen_inference_graph.xml"
LINUX_CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
WIN10_CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
# index of cat items used by the model
CAT_IDX   = 17
# confidence for detected bounding boxes
CONFIDENCE = 0.5

# PiCamera definition
INPUT_PICAM_WIDTH = 640
INPUT_PICAM_HEIGHT = 480

def get_cpu_extension():
    '''
    Try to guess which CPU extension to use
    '''
    if os.path.isfile(LINUX_CPU_EXTENSION):
        return LINUX_CPU_EXTENSION
    else:
        if os.path.isfile(WIN10_CPU_EXTENSION):
            return WIN10_CPU_EXTENSION
        else:
            sys.exit('CPU extension not found')

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    z_desc = "The forbidden zone"
    i_desc = "The location of the input video file"
    d_desc = "The device name, if not 'CPU'"
    s_desc = "Show forbidden zone in output, True or False"
    c_desc = "The path to the CPU extension, if not found"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-z", help=z_desc)
    optional.add_argument("-i", help=i_desc)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-s", help=s_desc, default=False)
    optional.add_argument("-c", help=c_desc, default=get_cpu_extension())
    args = parser.parse_args()

    return args

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, *image.shape)
    return image

def add_text_to_bounding_box(image, text, x_min, y_min, color):
    '''
    Add the text label to the bounding box
    '''
    labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x_min, y_min), (x_min + labelSize[0][0] + 10, y_min - labelSize[0][1] - 10), color, cv2.FILLED)
    image = cv2.putText(image, text, (x_min + 5, y_min - labelSize[0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

def create_output_image(image, output, forbidden_zone, width, height):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected
    # bounding boxes. For each detection, the description has the format:
    # [image_id, label, conf, x_min, y_min, x_max, y_max]
    # image_id - ID of the image in the batch
    # label - predicted class ID
    # conf - confidence for the predicted class
    # (x_min, y_min) - coordinates of the top left bounding box corner
    # (x_max, y_max) - coordinates of the bottom right bounding box corner.
    thickness = 1 # in pixels
    for bounding_box in output[0][0]:
        conf = bounding_box[2]
        if conf >= CONFIDENCE and CAT_IDX == int(bounding_box[1]):
            x_min = int(bounding_box[3] * width)
            y_min = int(bounding_box[4] * height)
            x_max = int(bounding_box[5] * width)
            y_max = int(bounding_box[6] * height)
            # calculation of the center of the bounding box
            center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            # choose color : red if center inside forbidden zone, green if outside
            color = (0, 255, 0)  # green
            label = 'good cat'
            if cv2.pointPolygonTest(forbidden_zone, center, False) > 0:
                color = (0, 0, 255) #red
                label = 'bad cat'
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            image = add_text_to_bounding_box(image, label, x_min, y_min, color)
    return image

def infer_on_video(args):
    '''
    Performs inference on video - main method
    '''
    # Test and parse the forbidden zone array
    if len(args.z) == 0:
        sys.exit('Error : forbidden zone not set')
    forbidden_zone = np.array(json.loads(args.z)).reshape((-1,1,2))
    if len(forbidden_zone) < 3: # a polygon should have at least 3 sides
        sys.exit('Error : forbidden zone format should be like "[[x1,y1],[x2,y2],[x3,y3]]"')

    # Test the "-i" argument
    is_reading_from_file = (args.i != None)
    if is_reading_from_file:
        if not os.path.isfile(args.i):
            sys.exit('Error : file "{0}" not found'.format(args.i))
        # Get and open video capture from file
        cap = cv2.VideoCapture(args.i)
        vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.open(args.i)
        # Grab the shape of the input
        width = int(cap.get(3))
        height = int(cap.get(4))
        # Create a video writer for the output video
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')` on Mac, and `0x00000021` on Linux
        # fourcc = cv2.VideoWriter_fourcc(*'x264')
        output_file = "output-" + str(args.i.rsplit(os.path.sep, 1)[-1])
        print("Writing output to ", output_file)
        out = cv2.VideoWriter(output_file, 0x7634706d, 30, (width, height))
    else:
        import time
        import imutils
        from imutils.video import VideoStream
        cap = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)

    is_drawing_forbidden_zone = (args.s == 'True')
    if is_drawing_forbidden_zone:
        print("Drawing forbidden zone on frames")

    ### Load the network model into the IE
    print("Load the network model into the IE")
    plugin = IECore()
    plugin.add_extension(args.c, args.d)
    network = IENetwork(model=MODEL, weights=os.path.splitext(MODEL)[0] + ".bin")
    exec_network = plugin.load_network(network, args.d)
    input_blob = next(iter(network.inputs))
    input_shape = network.inputs[input_blob].shape

    # If reading from file, init progress bar
    if is_reading_from_file:
        pbar = tqdm(total=vid_length)

    # Process frames until the video ends (if reading a file), or process is exited
    while not is_reading_from_file or cap.isOpened():

        if is_reading_from_file:
            pbar.update(1)
            flag, frame = cap.read()
            if not flag:
                break
        else:
            frame = cap.read()
            frame = imutils.resize(frame, width=INPUT_PICAM_WIDTH)

        key_pressed = cv2.waitKey(60)

        preprocessed_frame = preprocessing(frame, input_shape[2], input_shape[3])
        # Perform inference on the frame
        exec_network.start_async(request_id=0, inputs={input_blob: preprocessed_frame})

        if exec_network.requests[0].wait(-1) == 0:
            # Get the output of inference
            output = exec_network.requests[0].outputs
            # Show the forbidden zone if -s argument is True
            if is_drawing_forbidden_zone:
                cv2.polylines(frame, [forbidden_zone], True, (255, 0, 0))
            frame = create_output_image(frame, output['DetectionOutput'], forbidden_zone, width, height)

        # Write out the frame in the video
        out.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Starting")
    print("OpenCV version :  {0}".format(cv2.__version__))
    args = get_args()
    infer_on_video(args)

if __name__ == "__main__":
    main()
