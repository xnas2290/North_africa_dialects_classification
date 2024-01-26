import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio
import json

#importing the model
model = models.load_model("north_africa_dialects_classification.h5")
prob_model = models.Sequential([model,tf.keras.layers.Softmax()])


#importing the tokenizer
with open('tokenizer.json', "r") as json_file:
    loaded_word_index = json.load(json_file)
tokenizer = Tokenizer()
tokenizer.word_index = loaded_word_index

#classification function
def classify_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=300) 
    prediction = prob_model.predict(padded_sequences)[0]
    
    result = {
        'Algeria': prediction[0],
        'Egypt': prediction[1],
        'Morocco': prediction[2],
        'Tunisia': prediction[3],
    }
    
    return result

# Gradio Interface
iface = gradio.Interface(
    fn=classify_text,
    inputs=gradio.Textbox(),
    outputs=gradio.Label(num_top_classes=4),
    live=True,
    title="North africa dialects Classifier",
    description="Enter a text, and the model will classify it."
)

iface.launch(share=True)