import librosa
import opuspy
import numpy as np
import matplotlib.pyplot as plt


# Generate a dummy waveform for testing
# Load a .wav file with librosa
input_path = (
    "AC-00006_interactive_en_TLM-103_region_50000-60000.wav"  # Path to your .wav file
)
waveform, sample_rate = librosa.load(
    input_path, sr=48000, mono=True
)  # Load as stereo if available

# Ensure the waveform has int16 format expected by opuspy
waveform_int16 = (waveform * np.iinfo(np.int16).max).astype(np.int16)

# If waveform is mono, add a channel dimension to make it [time, channels]
waveform_tc = waveform_int16[:, np.newaxis]


output_path = "output.wav"
opuspy.write(
    path=output_path,
    waveform_tc=waveform_tc,
    sample_rate=sample_rate,
    bitrate=16000,
    signal_type=2,
    encoder_complexity=5,
    application=1,
    packet_loss_pct=0,
)

print(f"Audio saved to {output_path}")

# Test the `write` function
compr_wavs = []
for prob in [0.0, 0.0]:
    # Test the `read` function
    waveform_loaded, original_sample_rate = opuspy.read_with_packet_loss(
        output_path, loss_probability=prob
    )
    print(
        f"Loaded audio shape: {waveform_loaded.shape}, original sample rate: {original_sample_rate}"
    )
    compr_wavs.append(waveform_loaded)

n_fft = 1024
hop_length = 256
# create plot
fig, ax = plt.subplots(1, len(compr_wavs))
# make a spectrogram in db scale from wave forms
for audio, axs in zip(compr_wavs, ax):
    spec_db = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                audio.astype(np.float16).squeeze(), n_fft=n_fft, hop_length=hop_length
            )
        ),
        ref=np.max,
    )

    img = librosa.display.specshow(
        spec_db,
        y_axis="linear",
        x_axis="time",
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        ax=axs,
    )

plt.show()
