import audiofile
import os
from .utils import create_model
from .test import process as test_process
from pydub import AudioSegment
import soundfile as sf
import io

# Constants
MINS_TO_MILSECS = 60000
SECS_TO_MILSECS = 1000


# Functions
def stamp_to_int(time_stamp:str) -> int:
    minutes, seconds, milliseconds = [int(i) for i in time_stamp.split(":")]
    total = (MINS_TO_MILSECS * minutes) + (SECS_TO_MILSECS * seconds) + milliseconds
    return total


def int_to_stamp(time_int:int) -> str:
	minutes = int(time_int / MINS_TO_MILSECS)
	time_int -= minutes * MINS_TO_MILSECS
	seconds = int(time_int / SECS_TO_MILSECS)
	time_int -= seconds * SECS_TO_MILSECS
	milliseconds = time_int
	output = "{0:02d}:{1:02d}:{2:03d}".format(minutes, seconds, milliseconds)
	return output


# def process(transcript, audio_segment : sf.SoundFile):
def process(transcript, audio_segment : AudioSegment):
	timestamps = list(filter(None, transcript.split("\r\n")))
	speaker_dict = {}
	timestamps = [i.strip().split(", ") for i in timestamps]
	# Split data into timestamps and add speaker to dict if not present
	for segment in timestamps:
		timestamps[timestamps.index(segment)] = [stamp_to_int(i) if(":" in i) else i for i in segment]
		if(segment[-1] not in speaker_dict):
			speaker_dict.update({segment[-1]:[]})
	
	model = create_model()
	model.load_weights("gender_recognition\\results\\model.h5")

	count = 1
	# Processing audio chunks
	for snippet in timestamps:
		# Console output
		start, end, speaker = snippet[0], snippet[1], snippet[2]
		print("\nAudio {0}\nSpeaker {1}\n{2} - {3}".format(count, speaker, int_to_stamp(start), int_to_stamp(end)))
		# Create chunk to process
		audio_chunk = audio_segment[start:end]
		chunk_buffer = io.BytesIO()
		audio_chunk.export(chunk_buffer, format="mp3")
		# Test chunk, add to speaker dictionary
		temp_male_prob = test_process(chunk_buffer, model)[0]
		speaker_dict[speaker].append(temp_male_prob)
		count += 1

	output = ""
	# Average speaker probabilities and output to console
	for speaker in speaker_dict:
		total_male_prob = sum(speaker_dict[speaker])
		prob_instances = len(speaker_dict[speaker])
		avg_male_prob = (total_male_prob/prob_instances) * 100
		avg_female_prob = 100 - avg_male_prob
		output += "\nSpeaker: {0}\nInstances: {1}\nMale: {2:.2f}%\nFemale {3:.2f}%\n".format(
			speaker, prob_instances, avg_male_prob, avg_female_prob)
		
	return output
