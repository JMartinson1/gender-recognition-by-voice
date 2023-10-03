# Imports
import test
import utils
import os
from pydub import AudioSegment


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
	output = "{0:02d}:{1:02d}:{2}".format(minutes, seconds, milliseconds)
	return output


def main():
	with open("audio\\transcripts\\base_two_format2.srt", "r", encoding="utf-8") as file:
		timestamps = file.readlines()
	file.close()

	speaker_dict = {}
	timestamps = [i.strip().split(", ") for i in timestamps]
	for segment in timestamps:
		timestamps[timestamps.index(segment)] = [stamp_to_int(i) if(":" in i) else i for i in segment]
		if(segment[-1] not in speaker_dict):
			speaker_dict.update({segment[-1]:[]})
	
	audio_file = "audio\\full\\003NTWY_U1_CL.mp3"
	audio = AudioSegment.from_mp3(audio_file)
	model = utils.create_model()
	model.load_weights("results/model.h5")

	count = 1
	for snippet in timestamps:
		start, end, speaker = snippet[0], snippet[1], snippet[2]
		print("\nAudio {0}\nSpeaker {1}\n{2} - {3}".format(count, speaker, int_to_stamp(start), int_to_stamp(end)))
		audio_chunk = audio[start:end]
		chunk_name = "chunk{0}.wav".format(end)
		audio_chunk.export(chunk_name)
		temp_male_prob = test.process(chunk_name, model)[0]
		speaker_dict[speaker].append(temp_male_prob)
		os.remove(chunk_name)
		count += 1

	for speaker in speaker_dict:
		total_male_prob = sum(speaker_dict[speaker])
		prob_instances = len(speaker_dict[speaker])
		avg_male_prob = (total_male_prob/prob_instances) * 100
		avg_female_prob = 100 - avg_male_prob
		print("\nSpeaker: {0}\nInstances: {1}\nMale: {2:.2f}%\nFemale {3:.2f}%".
			format(speaker, prob_instances, avg_male_prob, avg_female_prob))
		
			
if(__name__ == '__main__'):
	main()
