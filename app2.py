"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from PIL import Image
import PIL
import threading

from sample_utils.turn import get_ice_servers

logger = logging.getLogger(__name__)

count=0
image_container={'img':None}
lock=threading.Lock()

@st.cache_data
def load_image(img_file_buffer):
    img = Image.open(img_file_buffer)
    return img

@st.cache_resource
def get_model():
    predict_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=False)
    predict_model.load_state_dict(torch.load("Model/model.pth"))
    predict_model.eval()
    return predict_model

def main():
    global count
    #Title
    st.set_page_config(page_title="Webapp for Braille Translator", layout="wide")
    st.title("Webapp for Braille Translator")

    #Sidebar
    st.sidebar.header('Predict')
    inputtype = ["Per letter", "Per words"]
    selected_type = st.sidebar.selectbox('Please select activity type to predict', inputtype)

    if selected_type == "Per letter":
        st.subheader("Per letter")
        acttype1 = ["Upload image", "Scan manually"]
        act_type_1 = st.sidebar.selectbox('Please select upload method to predict', acttype1)
        if act_type_1 == "Upload image":
            st.subheader("Upload image")
            st.write("Before you start uploading photo, make sure the dimension of the photo is fit.")
            st.image("./assets/example_letter.jpg", "Example Photo")
            img_file_buffer = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
            )
        
            predict_model = get_model()
            all_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'number', 'period', 'comma', 'colon', 'apostrophe', 'hyphen', 'semicolon', 'question', 'exclamation', 'capitalize'] 
            label = {key:value for key, value in enumerate(all_labels, start=0)}

            if img_file_buffer is not None:
                # View taken picture
                img = load_image(img_file_buffer)
                img = img.convert('RGB')
                # f.show()
                img = np.array(img)
                 # resize image to 28x28x3
                img = cv2.resize(img, (200, 200))
                # normalize to 0-1
                img = img.astype(np.float32)/255.0
                st.image(img, caption="Uploaded Image")

                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                img = img.permute(0,3,1,2)
                with torch.no_grad(): 
                    outputs = predict_model(img)
                    _, predicted = torch.max(outputs, 1)

                predicted_index = predicted.item()
                predicted_label = label[predicted_index]
                st.write('Predicted label:', predicted_label)
        
        elif act_type_1 == "Scan manually":
            st.subheader("Scan manually")
        
            predict_model = get_model()
            all_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'number', 'period', 'comma', 'colon', 'apostrophe', 'hyphen', 'semicolon', 'question', 'exclamation', 'capitalize'] 
            label = {key:value for key, value in enumerate(all_labels, start=0)}
            
            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                
                image = frame.to_ndarray(format="bgr24")
                with lock:
                    image_container['img']=image
                
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            
            ctx = webrtc_streamer(
                key="object-detection",
                rtc_configuration={
                    "iceServers": get_ice_servers(),
                    "iceTransportPolicy": "relay",
                },
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            p = st.empty()
            words=""
            while ctx.state.playing:
                with lock:
                    img=image_container['img']
                if img is None:
                    continue
                width, height, _ = img.shape
                img = img[int(height/2-100):int(height/2+100),int(width/2-50):int(width/2+150)]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # f.show()
                img = np.array(img)
                #  # resize image to 28x28x3
                img = cv2.resize(img, (200, 200))
                # # normalize to 0-1
                img = img.astype(np.float32)/255.0
                # # st.image(img, caption="Uploaded Image")

                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                img = img.permute(0,3,1,2)
                with torch.no_grad(): 
                    outputs = predict_model(img)
                    _, predicted = torch.max(outputs, 1)

                predicted_index = predicted.item()
                predicted_label = label[predicted_index]
                words+=predicted_label
                
                with p.container():
                    st.write("Predicted label:", predicted_label)
                
                # for debugging the frame taken
                # if count==100:
                #     cv2.imwrite("frame.jpg", img)
                #     count=0
                # count+=1

    elif selected_type == "Per words":
        st.subheader("Per words")
        
        acttype2 = ["Upload image", "Take a photo manually"]
        act_type_2 = st.sidebar.selectbox('Please select upload method to predict', acttype2)
        if act_type_2 == "Upload image":
            st.subheader("Upload image")
            st.write("Before you start uploading photo, make sure the dimension of the photo is fit.")
            st.image("./assets/example_words.jpg", "Example Photo")
            img_file_buffer = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
            )
        
            predict_model = get_model()
            all_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'number', 'period', 'comma', 'colon', 'apostrophe', 'hyphen', 'semicolon', 'question', 'exclamation', 'capitalize'] 
            label = {key:value for key, value in enumerate(all_labels, start=0)}

            if img_file_buffer is not None:
                image = load_image(img_file_buffer)
                image = image.convert('RGB')
                width, height = image.size
                num = round(width/height/0.75)
                w = width/num
                
                letters=[]
                for i in range (0,num):
                    cropped = image.crop((i*w,0,(i+1)*w,height))
                    st.image(cropped, "Cropped Image")
                    cropped = np.array(cropped)
                    cropped = cv2.resize(cropped, (200, 200))
                    cropped = cropped.astype(np.float32) / 255.0
                    cropped = torch.from_numpy(cropped[None, :, :, :])
                    cropped = cropped.permute(0, 3, 1, 2)
                    predicted_tensor = predict_model(cropped)
                    _, predicted_letter = torch.max(predicted_tensor, 1)
                    letters.append(chr(97 + predicted_letter))

                # predicted_index = letters.item()
                # predicted_label = label[letters]
                # print("Predicted Label:", predicted_label)
                # for x in range(len(letters)):
                words=""
                for x in letters:
                    words+=x[-1]
                st.write('Predicted label:', words)


        elif act_type_2 == "Take a photo manually":
            st.subheader("Take picture manually")
            with st.expander("See area of scan"):
                st.write("Before you take a photo, make sure each of Braille character is on the area of scan.")
                st.image("./assets/example_scan.png")

            predict_model = get_model()
            all_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'space', 'number', 'period', 'comma', 'colon', 'apostrophe', 'hyphen', 'semicolon', 'question', 'exclamation', 'capitalize'] 
            label = {key:value for key, value in enumerate(all_labels, start=0)}   

            img_file_buffer = st.camera_input("Take a picture")
            phone = st.checkbox('Using phone')
            if phone:
                if img_file_buffer is not None:
                    image = load_image(img_file_buffer)
                    image = image.convert('RGB')
                    width, height = image.size
                    st.write(image.size)
                    # image = image.rotate(90, PIL.Image.NEAREST, expand = 1)
                    st.image(image)
                    crop = image.crop((0,height/2-50,width,height/2+50))
                    width2, height2 = crop.size
                    num = round(width2/height2/0.75)
                    w = width2/num

                    letters=[]
                    for i in range (0,num):
                        cropped = crop.crop((i*w,0,(i+1)*w,height2))
                        st.image(cropped, "Cropped Image")
                        cropped = np.array(cropped)
                        cropped = cv2.resize(cropped, (200, 200))
                        cropped = cropped.astype(np.float32) / 255.0
                        cropped = torch.from_numpy(cropped[None, :, :, :])
                        cropped = cropped.permute(0, 3, 1, 2)
                        predicted_tensor = predict_model(cropped)
                        _, predicted_letter = torch.max(predicted_tensor, 1)
                        letters.append(chr(97 + predicted_letter))
                        st.write('Predicted label:', letters[i])
            else:            
                if img_file_buffer is not None:
                    image = load_image(img_file_buffer)
                    image = image.convert('RGB')
                    width, height = image.size
                    crop = image.crop((0,height/2-75,width,height/2+75))
                    width2, height2 = crop.size
                    num = round(width2/height2/0.75)
                    w = width2/num
                    
                    letters=[]
                    for i in range (0,num):
                        cropped = crop.crop((i*w,0,(i+1)*w,height2))
                        st.image(cropped, "Cropped Image")
                        cropped = np.array(cropped)
                        cropped = cv2.resize(cropped, (200, 200))
                        cropped = cropped.astype(np.float32) / 255.0
                        cropped = torch.from_numpy(cropped[None, :, :, :])
                        cropped = cropped.permute(0, 3, 1, 2)
                        predicted_tensor = predict_model(cropped)
                        _, predicted_letter = torch.max(predicted_tensor, 1)
                        letters.append(chr(97 + predicted_letter))
                        st.write('Predicted label:', letters[i])
                    
if __name__ == "__main__":
    main()