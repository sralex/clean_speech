# Clean Speech

This is a small AI desktop application demo with Tkinter + Keras.

The project consists in a LSTM network to create a IRM in order to separate the voice of an speaker from the original audio signal, the network is a similar approach to [https://arxiv.org/pdf/2002.11241.pdf](https://arxiv.org/pdf/2002.11241.pdf).  

Also this demo is similar to [https://github.com/sralex/speech_separation](https://github.com/sralex/speech_separation) but using tkinter.

The main purpose of the system is to separate the human voice from everything else, with few computational resources (currently 16mb), here you will find the project in deployment mode.

## How to use

## Prerequisites

* python >= 3.6
* pip3

## Instructions

Install using pip
```
pip install git+https://github.com/sralex/clean_speech.git
```

Then:

```
clean_speech
```

Demo:<br/>
![image](https://github.com/sralex/clean_speech/blob/main/demo.png)
