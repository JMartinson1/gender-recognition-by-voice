import io

from pydub import AudioSegment

from .test import process as test_process
from .utils import create_model

# Constants
MINS_TO_MILSECS = 60000
SECS_TO_MILSECS = 1000


# Functions
def stamp_to_int(time_stamp: str) -> int:
    minutes, seconds, milliseconds = [int(i) for i in time_stamp.split(":")]
    total = (MINS_TO_MILSECS * minutes) + (SECS_TO_MILSECS * seconds) + milliseconds
    return total


def int_to_stamp(time_int: int) -> str:
    minutes = int(time_int / MINS_TO_MILSECS)
    time_int -= minutes * MINS_TO_MILSECS
    seconds = int(time_int / SECS_TO_MILSECS)
    time_int -= seconds * SECS_TO_MILSECS
    milliseconds = time_int
    stamp_output = "{0:02d}:{1:02d}:{2:03d}".format(minutes, seconds, milliseconds)
    return stamp_output


def process(transcript: str, audio: AudioSegment) -> str:
    # Process transcript for timestamps and speaker data
    timestamps = list(filter(None, transcript.split("\r\n")))
    speaker_dict: dict = {}
    timestamps_list: list = [i.strip().split(", ") for i in timestamps]
    for segment in timestamps_list:
        timestamps_list[timestamps_list.index(segment)] = [stamp_to_int(i) if (":" in i) else i for i in segment]
        if segment[-1] not in speaker_dict:
            speaker_dict.update({segment[-1]: []})

    # Load model
    model = create_model()
    model.load_weights("gender_recognition\\results\\model.h5")

    # Processing audio chunks
    for snippet in timestamps_list:
        # Console output
        start = int(snippet[0])
        end = int(snippet[1])
        speaker = str(snippet[2])
        # Create chunk -> bytes to process
        audio_chunk = audio[start:end]
        chunk_buffer = io.BytesIO()
        audio_chunk.export(chunk_buffer, format="mp3")
        # Test chunk, add probability to speaker_dict
        temp_male_prob = test_process(chunk_buffer, model)[0]
        speaker_dict[speaker].append(temp_male_prob)

    # Average speaker probabilities and string output
    output = ""
    for speaker in speaker_dict:
        total_male_prob = sum(speaker_dict[speaker])
        prob_instances = len(speaker_dict[speaker])
        avg_male_prob = (total_male_prob / prob_instances) * 100
        avg_female_prob = 100 - avg_male_prob
        output += "\nSpeaker: {0}\nInstances: {1}\nMale: {2:.2f}%\nFemale {3:.2f}%\n".format(
            speaker, prob_instances, avg_male_prob, avg_female_prob
        )
    return output
