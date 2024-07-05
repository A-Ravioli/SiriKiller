import tkinter as tk
from tkinter import PhotoImage
import speech_recognition as sr
import pyttsx3
from ollama import Ollama
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Initialize Ollama with the quantized LLaMA model
ollama = Ollama(model_name="quantized-llama3")

# Load the GPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Use the speech recognition engine to listen to the microphone
def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}"
            )
        return ""


def generate_response(input_text, model=model, tokenizer=tokenizer):
    if torch.cuda.is_available():
        model = model.cuda()
    if model == "gpt2":
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Chatbot response: " + response_text)
        return response_text
    else:
        response = ollama.complete(prompt=input_text, max_tokens=50)
        response_text = response["choices"][0]["text"].strip()
        print("Chatbot response: " + response_text)
        return response_text


def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


def on_button_press(event):
    global recording
    recording = True
    while recording:
        input_text = speech_to_text()
        if input_text:
            response_text = generate_response(input_text)
            text_to_speech(response_text)


def on_button_release(event):
    global recording
    recording = False


# Initialize the GUI application
app = tk.Tk()
app.title("Voice Chatbot")
app.geometry("400x400")

# Create a canvas to draw the button
canvas = tk.Canvas(app, width=400, height=400, bg="white")
canvas.pack()

# Load the microphone icon image
mic_icon = PhotoImage(file="mic_icon.png")
mic_icon = mic_icon.subsample(3, 3)  # Adjust the size if necessary

# Draw the red circular button
button_radius = 50
button_x = 200
button_y = 200
canvas.create_oval(
    button_x - button_radius,
    button_y - button_radius,
    button_x + button_radius,
    button_y + button_radius,
    fill="red",
    outline="",
)

# Add the microphone icon to the center of the button
canvas.create_image(button_x, button_y, image=mic_icon)

# Create a transparent button over the red circle to handle the press event
button = tk.Button(
    app,
    text="",
    width=button_radius * 2,
    height=button_radius * 2,
    bg="red",
    activebackground="red",
    relief="flat",
    bd=0,
)
button_window = canvas.create_window(button_x, button_y, window=button, anchor="center")

# Bind the button press and release events
button.bind("<ButtonPress-1>", on_button_press)
button.bind("<ButtonRelease-1>", on_button_release)

# Start the Tkinter event loop
app.mainloop()
