import tkinter as tk
from tkinter import filedialog, ttk
import soundfile as sf
import tensorflow as tf
import numpy as np
import os
import keras
import scipy
from tkinter import messagebox
from keras.models import model_from_json

from clean_speech import feature_extractor, convert_to_audiowave


SR = 16000

def clean(data,progress_f = None):

    magnitudes, complex_ = feature_extractor(data)

    real_length = len(data)

    predictions = predict(magnitudes,progress_f)

    return convert_to_audiowave(predictions,complex_)[0:real_length]

def check_sr(data,current_sr,target_sr):

    if current_sr != target_sr:
        secs = len(data)/float(current_sr) # Number of seconds in signal X
        samps = int(secs*target_sr)     # Number of samples to downsample
        data = scipy.signal.resample(data, samps)

    return data

def c_speech(target_sr,filename,save_filename,progress_f = None):

    data, current_sr = sf.read(filename)

    data = check_sr(data, current_sr, target_sr)

    if len(data.shape) <= 2:

        if len(data.shape) == 1:

            data = clean(data,progress_f)

        else:
            data_list = []

            for i in range(data.shape[1]):
            
                data_list.append(clean(data[:,0],progress_f)[:,np.newaxis])

            data = np.concatenate(data_list)

    sf.write(save_filename,data,target_sr)

def predict(magnitudes ,progress_f = None):
    
    predictions = np.zeros(magnitudes.shape)
    
    total_features = len(magnitudes)

    for i in range(magnitudes.shape[0]):
        
        pred = model.predict(magnitudes[i:i+1])
        
        predictions[i,:,:,:] = pred[0,:,:,0:1]

        if not progress_f is None:
            progress_f(i,total_features)

    return predictions

def progress_tk_f(y,total_features):
    progress['value'] = int((y / float(total_features)) * 100)
    message_str.set("Cleaning ... {}%".format(int((y / float(total_features)) * 100)))
    master.update_idletasks() 

def compile_model():
    json_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(os.path.abspath(os.path.dirname(__file__)),'test.h5'))

    model.compile(
        optimizer= "adam",
        loss= "mse")

    return model 

def OpenFile(event=None):
    filename = filedialog.askopenfilename(filetypes=(("Audio","*.flag"),("Audio","*.wav")))
    e1_str.set( filename )

def SaveFile(event=None):
    filename = filedialog.asksaveasfilename(filetypes=(("Audio","*.flag"),("Audio","*.wav")))
    e2_str.set(filename)

def Predict(event=None):
    if(e1_str.get()=="" or e2_str.get()==""):
        messagebox.showinfo(title="Validation error", message="source file and output file must be selected")
        return
        
    c_speech(SR, e1_str.get(),e2_str.get(),progress_tk_f)
    progress['value'] = 100
    message_str.set("Task done!")

model = compile_model()

master=tk.Tk()

master.resizable(False, False)


master.title("Clean speech!")

label_input = tk.Label(master, text="source file")
label_input.grid(row=1,column=0,sticky='nesw')
label_output = tk.Label(master, text="output file")
label_output.grid(row=2,column=0,sticky='nesw')

message_str = tk.StringVar()
message_str.set("select the source file and the output file")
message = tk.Label(master, textvariable=message_str)
message.grid(row=3,column=0,sticky='nesw',columnspan=2)



button1=tk.Button(master, text="Open", command=OpenFile)
button1.grid(row=1,column=2,sticky='nesw')

button2=tk.Button(master, text="Open", command=SaveFile)
button2.grid(row=2,column=2,sticky='nesw')

button3=tk.Button(master, text="Convert",command = Predict)
button3.grid(row=3,column=2,sticky='nesw')


progress = ttk.Progressbar(master, orient=tk.HORIZONTAL,length=100,mode='determinate')


e1_str = tk.StringVar()
e2_str = tk.StringVar()

e1 = tk.Entry(master,textvariable=e1_str,state=tk.DISABLED)
e2 = tk.Entry(master,textvariable=e2_str,state=tk.DISABLED)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

progress.grid(row=4,column=0,columnspan=3,sticky='nesw')

master.mainloop()