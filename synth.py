from pathlib import Path
import argparse
import re
import os
from typing import Dict, List, Optional

from simpleaudio import Audio
from nltk.corpus import cmudict
import numpy as np


class Synth:
    def __init__(self, args: dict) -> None:
        self.all_diphones = self.load_diphone_data(args.diphones)
        self.comma_silence_time = 0.2  # unit: second
        self.period_silence_time = 0.4  # unit: second
        self.emphasis_flag = False  # a flag used for emphasis function
        self.emphasis_scale = 2  # the scale of emphasis (two times)
        self.crossfade_time = 0.01  # unit: second
        self.crossfade = args.crossfade
        self.reverse = args.reverse

    # reverse in "signal" way: switch the waveform signal for the whole synthetic utterance back to front
    @staticmethod
    def reverse_signal_way(audio: Audio) -> Audio:
        audio.data = audio.data[::-1]
        return audio

    # load the diphone data from the wav_folder, and generate an dictionary for all the diphones
    def load_diphone_data(self, wav_folder: str) -> Dict[str, str]:
        self.all_diphones = {}  # an empty dictionary for storing all the diphones and its corresponding .wav files
        # get all path of all the diphone files
        self.all_diphone_wav_files = (str(item) for item in Path(wav_folder).glob('*.wav') if item.is_file())
        # check if there is wav file in the list
        if not self.all_diphone_wav_files:
            print("there is no wav file in the {}".format(wav_folder))
        # for every .wav file, extract the real file name without path (e.g. diphones/) and the file suffix (e.g. .wav)
        for wav_file in self.all_diphone_wav_files:
            # get the file name first (e.g. xxx.wav)
            file_name_without_path = os.path.split(wav_file)[1]
            # extract the file name before ".wav"
            file_name = re.findall(r'(.+?)\.wav', file_name_without_path)
            # put the file name and wav file path together in the dictionary, file name as key, and the path as value
            self.all_diphones[file_name[0]] = wav_file
        
        # get the first file path in the all diphones dictionary and load it
        first_diphone_path = (list(self.all_diphones.values())[0])
        tmp_audio = Audio()  # initial an instance of class Audio
        # load the first diphone file
        tmp_audio.load(first_diphone_path)
        # in order to get the rate and nptype for later works
        self.rate = tmp_audio.rate
        self.nptype = tmp_audio.nptype
        del tmp_audio
        
        return self.all_diphones

    # generate an output audio of a diphone sequence with diphone files
    def get_output_audio_of_diphone_seq(self, diphone_seq_list: List[str]) -> Audio:
        output_audio = Audio()  # initial an instance of class Audio
        # create an empty np array for later stroring audio data
        diphone_seq_data = np.array([], dtype = self.nptype)
        # for every diphone, get the corresponding file path
        # load it and store to the diphone data
        # insert silence time for "," and "."
        for diphone in diphone_seq_list:
            # for "," and the punctuation sign "." (actually include ".", ":", "?", "!")
            # insert a corresponding silence array data
            if diphone == ',':
                # calculate the silence array length = silence time * rate
                array_len = int(np.floor(self.rate * self.comma_silence_time))  
                array = np.zeros(array_len, dtype=self.nptype)  # initial the zero np array for silence
                diphone_seq_data = np.concatenate((diphone_seq_data, array))  # add the array to diphone sequence data
            elif diphone == '.':
                array_len = int(np.floor(self.rate * self.period_silence_time))
                array = np.zeros(array_len, dtype=self.nptype)
                diphone_seq_data = np.concatenate((diphone_seq_data, array))
            # for emphasis sign "{" and "}", the switch of emphasis will accordingly turn on or off
            elif diphone == '{':
                self.emphasis_flag = True
            elif diphone == '}':
                self.emphasis_flag = False
            else:
                # convert each diphone (except "," "." "{" "}") into lower case first
                # since the wav file names are in lower case
                diphone = diphone.lower()
                try:
                    # get the file path of the diphone
                    diphone_file_path = self.all_diphones[diphone]
                except KeyError:
                    print('cannot find the wav file of "{}".'.format(diphone))
                else:
                    output_audio.load(diphone_file_path)  # load the corresponding diphone file path
                    # if the switch of emphasis is on, increase the loudness by emphasis_scale times
                    if self.emphasis_flag:
                        output_audio.data *= self.emphasis_scale 
                    # if the cross-fading is required, call the smoother_audio_concatenation function and get the processed diphone sequence data
                    if self.crossfade:
                        diphone_seq_data = self.smoother_audio_concatenation(diphone_seq_data, output_audio.data)
                    # else just put the data of the output_audio to the end of diphone sequence data
                    else:
                        diphone_seq_data = np.concatenate((diphone_seq_data, output_audio.data))

        # assign the data concatenated to the output audio
        output_audio.data = diphone_seq_data
        output_audio.data.dtype = self.nptype  # ensure the data type is the same as the nptype

        # if the user choose to reverse in "signal" way, call the reverse_signal function
        if self.reverse == 'signal':
            output_audio = self.reverse_signal_way(output_audio)  # then assign it to the output_audio

        return output_audio

    # smooth the audio concatenation by cross-fading between adjacent diphones using cross_fading_time overlap
    def smoother_audio_concatenation(self, audio_data_seq: np.ndarray, audio_data_add: np.ndarray) -> np.ndarray:
        # calculate the length of cross fading
        cross_fading_len = int(np.floor(self.crossfade_time * self.rate))
        # initial two arrays for cross-fading
        process_array_start = np.linspace(0, 1, cross_fading_len)
        process_array_end = np.linspace(1, 0, cross_fading_len)
        # use the cross-fading arrays to process the audio_data_add
        audio_data_add[-cross_fading_len:] = \
            (audio_data_add[-cross_fading_len:] * process_array_start).astype(self.nptype)
        audio_data_add[:cross_fading_len] = \
            (audio_data_add[:cross_fading_len] * process_array_end).astype(self.nptype)
        # if the length of the audio data sequence is 0， or say, if it is the data of the first diphone audio
        # directly put the processed audio_data_add to the audio_data_seq
        if len(audio_data_seq) == 0:
            concatenated_audio_data_seq = np.concatenate((audio_data_seq, audio_data_add))
        # if not, concatenate the two audio data with overlapping
        else:
            audio_data_seq[-cross_fading_len:] = \
                (audio_data_seq[-cross_fading_len:] + audio_data_add[:cross_fading_len]).astype(self.nptype)
            # put the rest of the audio_data_add to the audio data sequence
            concatenated_audio_data_seq = np.concatenate((audio_data_seq, audio_data_add[cross_fading_len:]))
        return concatenated_audio_data_seq


class Utterance:
    def __init__(self, phrase: str, spell: bool=False, reverse: Optional[str]=None) -> None:
        # normalise the input phrase and get a straight forward sequence of words
        self.lower_phrase = phrase.lower()  # convert the input phrase to lower case

        # deal with the punctuation
        # add a space before and after "," and "." for further spliting
        self.puncsign_phrase = re.sub(r',', ' , ', self.lower_phrase)
        self.puncsign_phrase = re.sub(r'\.', ' . ', self.puncsign_phrase)
        # add a space before and after "{" and "}" (emphasis sign) for further spliting
        # this step is for further dealing with the emphasis
        self.puncsign_phrase = re.sub(r'{', ' { ', self.puncsign_phrase)
        self.puncsign_phrase = re.sub(r'}', ' } ', self.puncsign_phrase)
        # change the ":", "?", "!" to "." since they will have same silence time
        # for further dealing with the silence time for these punctuation
        self.puncsign_phrase = re.sub(r'[:?!]', ' . ', self.puncsign_phrase)
        # ignore other punctuations, substitue all of them with spaces
        # remain " ' ", since some of the words have and can be pronunced through cmudict
        self.puncsign_phrase = re.sub(r"[^\w,.'{}]", ' ', self.puncsign_phrase)  
        self.seq_words = self.puncsign_phrase.split()  # split it to a string list

        # if the user input "-s" or "--spell", convert the word sequence to a sequence of letters
        if spell:
            print("You choose to spell it out.")
            self.seq_letters = []  # an empty list to store letter sequence
            # get the letters sequence from each word
            for word in self.seq_words:
                self.seq_letters.extend(list(word))
            # then assign the letter sequence to the seq_words
            # so that the following get_phone_seq and get_diphone_seq functions can work
            self.seq_words = self.seq_letters

        # check if the user input "-r" or "--reverse" and the reverse way
        # if yes, call the reverse_way function
        self.reverse = reverse
        if reverse is not None:
            print("The reverse way you choose is: {}".format(reverse))
            self.seq_words = self.reverse_ways(reverse)

    # get the words sequence if the user ask for reverse
    def reverse_ways(self, reverse_way: str) -> List[str]:
        # for the reverse way "words"
        if reverse_way == 'words':
            # reverse the order of the words that will be synthesised
            # assign it to seq_words for following function
            self.seq_words = self.seq_words[::-1]
            # if the input ask for emphasis
            # swap the "{" and "}"
            seq_words_tmp = ['*' if i == '{' else i for i in self.seq_words]
            seq_words_tmp = ['{' if i == '}' else i for i in seq_words_tmp]
            seq_words_tmp = ['}' if i == '*' else i for i in seq_words_tmp]
            self.seq_words = seq_words_tmp
            return self.seq_words
        # for other reverse ways, return the seq_words without changes
        else:
            return self.seq_words

    # get the phone sequence of the input phrase
    def get_phone_seq(self) -> List[str]:
        alphabet = cmudict.dict()  # a pronunciation lexicon provided as a part of NLTK
        self.phone_seq_original = []  # an empty list for the original phones caught from cmudict
        # create an empty list to save the words that is not in cmudict and cannot be pronounced
        self.words_cannot_pronunced = []  

        for word in self.seq_words:
            # check if the word is in cmudict
            # if yes, get the pronunciation and add it to the phone list
            # for ",", ".", "{" and "}", remain
            # if the word is not in cmudict, add it to the words_cannot_pronunced list
            if word in alphabet:
                self.phone_seq_original.extend(alphabet[word][0])
            elif word in [',', '.']:
                self.phone_seq_original.extend(['PAU', word, 'PAU'])  # add "PAU" before and after a punctuation
            elif word in ['{', '}']:
                self.phone_seq_original.extend(word)
            else:
                self.words_cannot_pronunced.append(word)
        # if there are some words cannot be pronunced in the phrase, tell the user
        if self.words_cannot_pronunced:
            print('The word "{}" cannot be pronounced because it is not in the cmudict.'
                  .format(self.words_cannot_pronunced))

        # the original phones provided by cmudict have numbers
        # so try to delete the numbers in order to match the names of diphone wav files
        self.phone_seq_without_num = []  # an empty list for storing the phones without numbers
        for ori_phone in self.phone_seq_original:
            self.phone_seq_without_num.append(re.sub(r'\d', '', ori_phone))

        # utterance should always start and end with an silence phone
        # so add "PAU" (a label for the short pause or silence phone) to the first and the end of the phone sequence
        self.phone_seq_without_num.insert(0, 'PAU')
        # but if the last phone is "PAU", not add an extra "PAU"
        # since a "PAU" has already added if there is a punctuation at the end
        if self.phone_seq_without_num[-1] != 'PAU':
            self.phone_seq_without_num.extend(['PAU'])
        self.phone_seq_with_pau = self.phone_seq_without_num  # then assign it to phone_seq_with_pau

        self.phone_seq = self.phone_seq_with_pau  # the final vision of the phone sequence

        # check the reverse way
        # if it is "phones", call the reverse_phones_way function
        if self.reverse == 'phones':
            self.phone_seq = self.reverse_phones_way(self.phone_seq)            

        return self.phone_seq

    # reverse in "phones" way: reverse the order of the phones that will be spoken for the whole utterance
    @staticmethod
    def reverse_phones_way(phone_seq: List[str]) -> List[str]:
        phone_seq = phone_seq[::-1]  # assign it to the phone_seq for following function
        # if the input ask for emphasis
        # swap the "{" and "}"
        phone_seq_tmp = ['*' if i == '{' else i for i in phone_seq]
        phone_seq_tmp = ['{' if i == '}' else i for i in phone_seq_tmp]
        phone_seq_tmp = ['}' if i == '*' else i for i in phone_seq_tmp]
        phone_seq = phone_seq_tmp
        return phone_seq

    # get the corresponding diphone sequence
    @staticmethod
    def get_diphone_seq(phone_seq: List[str]) -> List[str]:
        diphone_seq = []  # an empty list to store the diphones
        # for every phone in the phone seq list, use "-" to link the phone and its following phone
        # for ",", ".", "{" and "}", remain
        # for phone just before "{" and "}", use "-" to link the phone and the next phone
        # then store it to the diphone seq
        for num in range(len(phone_seq)-1):
            if (phone_seq[num+1] not in [',', '.', '{', '}']) & (phone_seq[num] not in [',', '.', '{', '}']):
                diphone_seq.append(phone_seq[num] + '-' + phone_seq[num+1])
            elif phone_seq[num] in [',', '.', '{', '}']:
                diphone_seq.append(phone_seq[num])
            elif phone_seq[num+1] in ['{', '}']:
                diphone_seq.append(phone_seq[num] + '-' + phone_seq[num+2])
        return diphone_seq
