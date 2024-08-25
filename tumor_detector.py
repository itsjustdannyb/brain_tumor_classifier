import gradio as gr
import numpy as np
from PIL import Image

# load the model
import torch
from torchvision import transforms

# the model
from model import Net

print("imports ok!")

def predictor(img):

    classes = {
        0: "no tumor",
        1: "tumor"
    }
    path = r"C:\Users\Daniel\Documents\brain_tumor_classifier\brain_tumor_model"
    net = Net()
    net.load_state_dict(torch.load(path))

    img_pil = Image.fromarray(img.astype('uint8'), 'RGB')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((125, 125))])
    img = transform(img_pil)
    img = img.unsqueeze(0)

    output = net(img)
    prediction = (output >= 0.5).float()

    return classes[prediction.item()]


demo = gr.Interface(fn=predictor, inputs=[gr.Image()], outputs=[gr.Textbox(label="The model thinks ...")], title="tumor_detector", description="This model predicts if an MRI brain scan has a tumor or not.")
demo.launch(share=True)