import argparse
import cv2
import os
import sys
from openvino.inference_engine import IENetwork, IECore, np

MODEL = "models/frozen_inference_graph.xml"
LINUX_CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
WIN10_CPU_EXTENSION = "C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll"
# index of cat items used by the model
CAT_IDX   = 17
TABLE_IDX =  67
# confidence for detected bounding boxes
CONFIDENCE = 0.5

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
    s_desc = "Show forbidden zone, True or False"
    c_desc = "The path to the CPU extension, if not found"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-z", help=z_desc)
    optional.add_argument("-i", help=i_desc)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-s", help=s_desc, default=False)
    optional.add_argument("-c", help=c_desc, default=LINUX_CPU_EXTENSION)
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
    print("inputimage= "+str(image))
    return image

def add_text_to_bounding_box(image, text, x_min, y_min, color):
    '''
    Add the text label to the bounding box
    '''
    labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x_min, y_min), (x_min + labelSize[0][0] + 10, y_min - labelSize[0][1] - 10), color, cv2.FILLED)
    image = cv2.putText(image, text, (x_min + 5, y_min - labelSize[0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

def create_output_image(image, output, r1, width, height):
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
   

    ######
    for bounding_box in output[0][0]:
        conf = bounding_box[2]
        if conf >= 0.4 and TABLE_IDX == int(bounding_box[1]):
            x_min1 = int(bounding_box[3] * width)
            y_min1 = int(bounding_box[4] * height)
            x_max1 = int(bounding_box[5] * width)
            y_max1 = int(bounding_box[6] * height)
            
            #r1= cv2.rectangle(x_min1+x_max1 , y_min1+y_max1, (255,0,0),1)
            r1 = np.array([[x_min1, y_min1], [x_min1, y_max1], [x_max1, y_min1], [x_max1, y_max1]], np.int32)
            #print("XXXXXXX")
           # print(r1)
            #print("XXXXXXX")
            # calculation of the center of the bounding box
            center = ((x_min1 + x_max1) / 2, (y_min1 + y_max1) / 2)
            # choose color : blue for table
            color = (255, 0, 0)  # Blue
            label = 'Table'
          
            cv2.rectangle(image, (x_min1, y_min1), (x_max1, y_max1), color, thickness)
            image = add_text_to_bounding_box(image, label, x_min1, y_min1, color)
            table_box = image
    #######
    
   
    for bounding_box in output[0][0]:
        conf = bounding_box[2]
        if conf >= CONFIDENCE and CAT_IDX == int(bounding_box[1]):
            x_min2 = int(bounding_box[3] * width)
            y_min2 = int(bounding_box[4] * height)
            x_max2 = int(bounding_box[5] * width)
            y_max2 = int(bounding_box[6] * height)
            
            r2= cv2.rectangle(x_min2+x_max2 , y_min2+y_max2, (255,0,0),1)
      
                
            # calculation of the center of the bounding box
            center = ((x_min2 + x_max2) / 2, (y_min2 + y_max2) / 2)
            # choose color : red if center inside forbidden zone, green if outside
            color = (0, 255, 0)  # green
            label = 'Good kitty'
            if cv2.pointPolygonTest(r1, center, False) > 0:
                color = (0, 0, 255) #red
                label = 'Bad kitty'
            cv2.rectangle(image, (x_min2, y_min2), (x_max2, y_max2), color, thickness)
            image = add_text_to_bounding_box(image, label, x_min2, y_min2, color)
    return image

##
    
##..

def infer_on_video(args,r1):
    '''
    Performs inference on video - main method
    '''
    ### TODO : Parse the forbidden zone array
    # forbidden_zone = np.fromstring(args.z, dtype=[('x',int),('y',int)], sep=',')
    #forbidden_zone = np.array([[0, 100], [470, 160], [260, 350], [0, 290]], np.int32).reshape((-1,1,2))
    #forbidden_zone = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32).reshape((-1,1,2))
    #
    ### Load the network model into the IE
    print("Load the network model into the IE")
    plugin = IECore()
    #print(str(IECore()))
    plugin.add_extension(args.c, args.d)
    network = IENetwork(model=MODEL, weights=os.path.splitext(MODEL)[0] + ".bin")
    exec_network = plugin.load_network(network, args.d)
    input_blob = next(iter(network.inputs))
    input_shape = network.inputs[input_blob].shape
    print(str(input_shape))
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    print(args.i)
    print(str(cap))
    

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    print (str(cap.get(4)))
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')` on Mac, and `0x00000021` on Linux
    # fourcc = cv2.VideoWriter_fourcc(*'x264')
    output_file = "output-" + str(args.i.rsplit(os.path.sep, 1)[-1])
    print("Writing output to ", output_file)
    
    #out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    #fourcc=cv2.VideoWriter_fourcc('F','M','P','4')
    
    #out = cv2.VideoWriter(output_file,fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))
    
    #print (str(fourcc))
    
    out = cv2.VideoWriter(output_file, 0x00000021, 30, (width,height))
    
    print(str(out))
    
    # Process frames until the video ends, or process is exited
    frame_count = 0;
    
    print(str(cap.isOpened()))
    cap.open(args.i)
    while cap.isOpened():
        # Read the next frame
       
        flag, frame = cap.read()
        if not flag:
            break
       
        key_pressed = cv2.waitKey(60)
        preprocessed_frame = preprocessing(frame, input_shape[2], input_shape[3])
        print("Perform inference on the frame")
        exec_network.start_async(request_id=0, inputs={input_blob: preprocessed_frame})
        
      
        if exec_network.requests[0].wait(-1) == 0:
            # Get the output of inference
            output = exec_network.requests[0].outputs
            # Show the forbidden zone if -s argument is True
            if args.s:
                cv2.polylines(frame, [args.z], True, (255, 0, 0))
            frame = create_output_image(frame, output['DetectionOutput'], r1, width, height)

        # Write a frame here for debug purpose
        # cv2.imwrite("frame" + str(frame_count) + ".png", frame)

        # Write out the frame in the video
        out.write(frame)

        # frame count
        frame_count = frame_count + 1

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
    r1 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32).reshape((-1,1,2))
    args = get_args()
    infer_on_video(args,r1)


if __name__ == "__main__":
    main()
