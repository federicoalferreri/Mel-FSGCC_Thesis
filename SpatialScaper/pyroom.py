import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import wavfile

import spatialscaper as ss
import pyroomacoustics as pra

# The desired reverberation time and dimensions of the room
rt60_tgt = 0.6  # seconds
room_dim = [15, 20, 3.5]  # meters


# the sampling frequency should match that of the room
fs = 24000


# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

pra.constants.set("octave_bands_keep_dc", True)

room = pra.ShoeBox(
      room_dim,
      fs=fs,
      materials=pra.Material(e_absorption),
      max_order=max_order,
      use_rand_ism=True,
      air_absorption=True,
  )

# place the source in the room
room.add_source([13.5, 2.73, 1.76], delay=0.5)
room.add_source([10.5, 6.73, 2.7], delay=0.5)
room.add_source([3.5, 9.73, 1.0], delay=0.5)
room.add_source([5.5, 14.73, 0.6], delay=0.5)
room.add_source([7.5, 17.73, 1.3], delay=0.5)

source_locs = np.array([
    np.array([13.5, 2.73, 1.76]),
    np.array([10.5, 17.73, 2.7]),
    np.array([3.5, 9.73, 1.0]),
    np.array([5.5, 14.73, 0.6]),
    np.array([7.5, 6.73, 1.3]),
])

'''n_mics = 32
y_start = 3  # starting point on the y-axes
y_step = 0.5  # Distance between microphones (0.5 m)
z_fixed = 1.2
x_fixed = 2.5

y_positions = y_start + np.arange(n_mics) * y_step
x_positions = np.full(n_mics, x_fixed)
z_positions = np.full(n_mics, z_fixed)

mic_locs = np.vstack([x_positions, y_positions, z_positions])'''
y_start = 3  # starting point on the y-axes
z_fixed = 1.2
x_fixed = 2.5

y_positions = np.array([9, 10, 11, 12])
x_positions = np.full(len(y_positions), x_fixed)
z_positions = np.full(len(y_positions), z_fixed)

mic_locs = np.vstack([x_positions, y_positions, z_positions])
print(mic_locs)

# finally place the array in the room
room.add_microphone_array(mic_locs)
fig, ax = room.plot()
# Run the simulation (this will also build the RIR automatically)
room.compute_rir()


def NEW_get_room_irs_xyz():
    return source_locs


def NEW_get_room_irs_wav_xyz():
    """
    Modified function to pad RIR data correctly to handle variable channel lengths.

    - Pads individual channels within each RIR before padding the RIRs themselves.
    - This ensures uniformity across all channels and prevents the ValueError.
    """
    room_rir_array = room.rir
    num_sources = len(room.sources)
    num_mics = len(room.mic_array)

    # Determine maximum RIR length
    max_len = max(len(rir[0]) for rir in room_rir_array)

    # Pad individual RIR channels
    padded_rir = np.zeros((num_sources, num_mics, max_len))
    for source_idx in range(num_sources):
        for mic_idx in range(num_mics):
            rir_len = len(room_rir_array[mic_idx][source_idx])
            # Adjust assignment to accommodate variable RIR lengths
            padded_rir[source_idx, mic_idx, :min(rir_len, max_len)] = room_rir_array[mic_idx][source_idx][
                                                                      :min(rir_len, max_len)]

    return padded_rir, 24000, source_locs

print(len(room.sources))
print(len(room.mic_array))

max_len = max(len(rir[0]) for rir in room.rir)
print(max_len)

# Constants
NSCAPES = 1  # Number of soundscapes to generate
FOREGROUND_DIR = "/nas/home/fferreri/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = None # Directory containing Room Impulse Response (RIR) files

ROOM = room  # Initial room setting, change according to available rooms listed below
FORMAT = "mic"  # Output format specifier
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape, customizable by the user
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "/nas/home/fferreri/baseline_codes/Dataset_tests"  # Directory to store the generated soundscapes
REF_DB = -65  # Reference decibel level for the background ambient noise. Try making this random too!
print(OUTPUT_DIR)
# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.


# Function to generate a soundscape
def generate_soundscape(index):
    track_name = f"fold5_room1_mix{index+1:03d}"
    # Initialize Scaper. 'max_event_overlap' controls the maximum number of overlapping sound events.
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        ROOM,
        max_event_overlap=3,
        speed_limit=2.0,  # in meters per second
    )
    ssc.ref_db = REF_DB

    ssc.get_room_irs_xyz = NEW_get_room_irs_xyz
    ssc.get_room_irs_wav_xyz = NEW_get_room_irs_wav_xyz

    # static ambient noise
    ssc.add_background()

    # Add a random number of foreground events, based on the specified mean and standard deviation.
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    n_events = n_events if n_events > 0 else 1  # n_events should be greater than zero

    for _ in range(n_events):
        ssc.add_event(label = ('choose', ['femaleSpeech', 'maleSpeech', 'telephone', 'laughter','domesticSounds', 'footsteps', 'music','musicInstrument']))  # randomly choosing and spatializing an FSD50K sound event

    audiofile = os.path.join(OUTPUT_DIR, "mic_dev", "mic", track_name)
    labelfile = os.path.join(OUTPUT_DIR, "metadata_dev", "labels", track_name)

    ssc.generate(audiofile, labelfile)



# Main loop for generating soundscapes
for iscape in range(NSCAPES):
    print(f"Generating soundscape: {iscape + 1}/{NSCAPES}")
    generate_soundscape(iscape)