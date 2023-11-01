# Text-to-Speech Synthesis using Diphone Concatenation

## Introduction
This is a basic text-to-speech synthesis application built in Python. It synthesizes speech using the method of diphone concatenation. Diphones are pairs of phonemes, and they are used here as the fundamental units for constructing larger sequences of speech.

By utilizing a set of pre-recorded diphone sound samples, this program can produce speech output for a given text input. Additionally, there are options to manipulate the output, such as controlling the volume, spelling out the input, reversing the speech in different modes, cross-fading between diphones for smoother transitions, and more.

The instruction of the project can be found in `./docs`

## Requirements
Ensure you have the following:
- Python 3.10 or plus
- The necessary diphone WAV files stored in a folder
- PyAudio==0.2.13
- nltk==3.8.1
- numpy==1.26.1

## Usage

run following command to find out the function of the script:
```bash
python main.py -h
```

### Basic Command
```bash
python <script_name>.py [phrase] --diphones [path_to_diphone_folder]
```
Replace `<script_name>.py` with the actual name of your Python script.

### Examples

1. Basic synthesis
    ```bash
    python main.py -p "A rose by any other name would smell as sweet"
    ```

2. Reverse speech by words/phones/signal
    ```bash
    python main.py -r words -p "A rose by any other name would smell as sweet"
    ```

3. Save output to file
    ```
    python main.py -o ./examples/rose.wav "A rose by any other name would smell as sweet"
    ```
