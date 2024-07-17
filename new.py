### importing required libraries
import torch
import cv2
import time
#import pytesseract
import re
import numpy as np
#import easyocr
import pandas as pd
import csv
import uuid
import os
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def save_results(text,region,csv_filename,folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    cv2.imwrite(os.path.join(folder_path, img_name), region)


### -------------------------------------- function to run detection ---------------------------------------------------------

def detectx(frame, model):
    frame = [frame]
    print("[INFO] Detecting...")
    results = model(frame)
    results.show()

    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    label_names = [f"Frame {i}" for i in range(1, len(labels) + 1)]

    model =  torch.hub.load('yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally
    classes = model.names  # class names in string format
    
    ppi = 92

    width = 1920
    height = 1080

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    width_cm = width / ppi * 2.54
    height_cm = height / ppi * 2.54

    x_cm =  x / ppi * 2.54
    y_cm = y / ppi * 2.54

    # Get the correct class name for each label
    class_names = []
    for label in labels:
        class_name = classes[int(label)]
        class_names.append(class_name)
    # Create a Pandas DataFrame with the labels, coordinates, and class names
    df = pd.DataFrame({
    "FRAMES": label_names,
    # "FRAME WIDTH(cm)" : width_cm,
    # "FRAME HEIGHT(cm)": height_cm,
    "DEFECT LOCATION((x)":(x_cm)*1000,
    "DEFECT LOCATION{y)":(y_cm)*1000,
    "WIDTH": coordinates[:, 2],
    "HEIGHT": coordinates[:, 3],
    "CLASSES": class_names
})

# calculate the WIDTH(Inches) and HEIGHT(Inches) columns using the WIDTH and HEIGHT columns
    df["WIDTH(Inches)"] = (df["WIDTH"]/25) * 10 
    df["HEIGHT(Inches)"] = (df["HEIGHT"]/15)* 10 

# calculate the SEVERITY column using the Excel formula
    df["SEVERITY"] = np.where((df["WIDTH(Inches)"].astype(int) > 3) | (df["HEIGHT(Inches)"].astype(int) > 3), "MAJOR", "MINOR")
    # Open the Excel file and append the DataFrame to it
    wb = load_workbook(filename='results.xlsx')
    ws = wb['Sheet1']

    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    wb.save('results.xlsx')

    return labels, coordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):
    """
    This function takes results, frame and classes
    results: contains labels and coordinates predicted by model on the given frame
    classes: contains the string labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    # Loop through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.10: # Threshold value for detection. Discard everything below this value.
            print(f"[INFO] Extracting Box coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) # BBox coordinates
            text_d = classes[int(labels[i])]

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # BBox

            # Add a label to the detected object
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            label = f"{text_d}"
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1 - 10), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, (255, 255, 255), thickness)

    return frame




### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format
    print(classes)




    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame,classes = classes)
        

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.

                break

    ### --------------- for detection on video --------------------
    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 1
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        
        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


# main(vid_path="./test_images/vid_1.mp4",vid_out="vid_1.mp4") ### for custom video
main(vid_path=1,vid_out="webcam_result.mp4") #### for webcam

# main(img_path="D:\\PROJECTS\\FABRICS\\1.jpeg") ## for image
            

