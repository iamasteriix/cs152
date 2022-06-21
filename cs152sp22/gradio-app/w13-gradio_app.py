#!/usr/bin/env python
"""
Run this file with:
GRADIO_SERVER_PORT=<port> python w13-gradio_app.py </path/to/model>

Where
- 8961 is your server port number
- ./resnet18-1.pkl is the path to the model you want to use for inference
"""

from fastai.vision.all import *
import gradio as gr


# Load the trained model
path = Path(sys.argv[1])
model = load_learner(path)


def classify(img):

    prediction = model.predict(img)
    label = prediction[0]
    label_index = prediction[1]
    probabilities = prediction[2]
    label_prob = probabilities[label_index]

    return f"Almost {label_prob*100:.1f}% certain that's {prediction[0]}."

title = "Let's play!"
website = "Gradio app" 

# iface = gr.Interface(fn=classify, inputs=gr.inputs.Image(), outputs="text", title=title, article=website, theme="dark").launch()

gr.Interface(fn=classify, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)
