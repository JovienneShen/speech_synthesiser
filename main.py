import argparse
import re
import os
import numpy as np

from synth import Synth, Utterance
from simpleaudio import Audio

# process and synthesise the phrase and output the audio
def process_phrase_to_output(phrase: str) -> Audio:
    # get the synthesised sequence of words
    utt = Utterance(phrase=phrase, reverse=args.reverse, spell=args.spell)
    # expand the word sequence to a phone sequence
    phone_seq = utt.get_phone_seq()
    # expand the phone sequence to a corresponding diphone sequence
    diphone_seq = utt.get_diphone_seq(phone_seq)
    # get the audio of the diphone sequence
    output_audio = diphone_synth.get_output_audio_of_diphone_seq(diphone_seq)
    volume_control(output_audio)  # check the volume control

    # if the user input '-p', then play the audio
    if args.play:
        output_audio.play()

    return output_audio

# save the audio as the name given by user
def save_audio(audio: Audio) -> None:
    save_filename = args.outfile
    # first check if the given filename for saving has a suffix ".wav"
    # if not, add the ".wav" to the filename
    if re.findall(r'[^.]+$', save_filename) != ["wav"]:
        save_filename += ".wav"
    print("Save it as {}".format(save_filename))
    audio.save(save_filename)
    

# control the volume to change the amplitude of the synthesised waveform
def volume_control(audio: Audio) -> Audio:
    if args.volume is not None:
        print("Control the volume to: {}".format(args.volume))
        # check if the input volume is an integer between 0 and 100
        if (args.volume >= 0) & (args.volume <= 100):
            rescale_factor = args.volume/100  # convert the input volume number to a number between 0 and 1
            # change the amplitude of the synthesised waveform
            audio.data = (audio.data * rescale_factor).astype(audio.nptype)
            return audio
        else:
            print("Please enter a volume number between 0 and 100.")


# process the input text (after --fromfile)
def process_from_file(text_file_name: str) -> Audio:
    # create an empty phrase list to temporarily store the sentence in the text file
    phrase_tmp = ''
    data_tmp = np.array([], dtype=diphone_synth.nptype)  # create an empty data array to store wav information
    # open the given file
    with open(text_file_name, 'r') as file_to_read:
        while True:  # a potentially forever loop
            line = file_to_read.readline()  # read the text line by line
            # if the text reach the end, process the phrase_tmp and break the loop
            if not line:
                if phrase_tmp != '':
                    audio_tmp = process_phrase_to_output(phrase_tmp)
                    data_tmp = np.concatenate((data_tmp, audio_tmp.data))
                break
            # find the position of ".", "!", ":" and "?" in each processing line
            period_iter = re.finditer(r"[.!?:]", line)
            index_period = [m.start(0) for m in period_iter]
            # if there is no such punctuation, put the whole line into phrase_tmp and process the next line
            if not index_period:
                phrase_tmp += line
            # if this line has such punctuation, put the phrase before the last punctuation into phrase_tmp
            else:
                phrase_tmp += line[:index_period[-1]+1]
                # from the phrase_tmp which contains one or several complete sentences, initial an audio
                audio_tmp = process_phrase_to_output(phrase_tmp)
                # extend the data_tmp by adding the wav data of the new phrase to its end
                data_tmp = np.concatenate((data_tmp, audio_tmp.data))
                # reset the phrase_tmp to the rest text of the line
                phrase_tmp = line[index_period[-1]+1:]
    audio_tmp.data = data_tmp
    return audio_tmp

# process the commandline and return args
def process_commandline():
    parser = argparse.ArgumentParser(
        description='A basic text-to-speech app that synthesises speech using diphone concatenation.')

    # basic synthesis arguments
    parser.add_argument('--diphones', default="./diphones",
                        help="Folder containing diphone wavs")
    parser.add_argument('--play', '-p', action="store_true", default=False,
                        help="Play the output audio")
    parser.add_argument('--outfile', '-o', action="store", dest="outfile",
                        help="Save the output audio to a file", default=None)
    parser.add_argument('phrase', nargs='?',
                        help="The phrase to be synthesised")

    # Arguments for extension tasks
    parser.add_argument('--volume', '-v', default=None, type=int,
                        help="An int between 0 and 100 representing the desired volume")
    parser.add_argument('--spell', '-s', action="store_true", default=False,
                        help="Spell the input text instead of pronouncing it normally")
    parser.add_argument('--reverse', '-r', action="store", default=None, choices=['words', 'phones', 'signal'],
                        help="Speak backwards in a mode specified by string argument: 'words', 'phones' or 'signal'")
    parser.add_argument('--fromfile', '-f', action="store", default=None,
                        help="Open file with given name and synthesise all text, which can be multiple sentences.")
    parser.add_argument('--crossfade', '-c', action="store_true", default=False,
                        help="Enable slightly smoother concatenation by cross-fading between diphone units")

    args = parser.parse_args()

    if (args.fromfile and args.phrase) or (not args.fromfile and not args.phrase):
        parser.error('Must supply either a phrase or "--fromfile" to synthesise (but not both)')

    return args   

if __name__ == "__main__":
    args = process_commandline()

    print(f'Will load wavs from: {args.diphones}')
    # first, check if the input wav_folder (after --diphones) exists
    if os.path.exists(args.diphones):
        # initial a Synth class
        diphone_synth = Synth(args)

        # if the input ask open a file with given name and synthesise all text
        if args.fromfile is not None:
            # first check if the input is a text file
            if re.findall(r'[^.]+$', args.fromfile) == ["txt"]:
                # check if the given file exists
                if os.path.isfile(args.fromfile):
                    print("Synthesise the text file: {}".format(args.fromfile))
                    out_put_audio = process_from_file(args.fromfile)
                    # if the user input '-o' and a filename, call the save_audio function
                    if args.outfile is not None:
                        save_audio(out_put_audio)
                else:
                    print('The given file "{}" does not exist.'.format(args.fromfile))
            else:
                print('Please provide a text file.')

        # else, synthesise the input phrase
        else:
            print(f'You printed: {args.phrase}')  # tell the user what is the input
            out_put_audio = process_phrase_to_output(args.phrase)
            # if the user input '-o' and a filename, call the save_audio function
            if args.outfile is not None:
                save_audio(out_put_audio)
    else:
        print("The directory of diphones does not exist.")