#include <iostream>
#include "opusenc.h"
#include "opusfile.h"
#include "opus.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <opus/opus.h>
#include <ogg/ogg.h>

namespace py = pybind11;

template<typename T>
py::array_t<T> MakeNpArray(std::vector<ssize_t> shape, T* data) {

    std::vector<ssize_t> strides(shape.size());
    size_t v = sizeof(T);
    size_t i = shape.size();
    while (i--) {
        strides[i] = v;
        v *= shape[i];
    }
    py::capsule free_when_done(data, [](void* f) {
        auto* foo = reinterpret_cast<T*>(f);
        delete[] foo;
    });
    return py::array_t<T>(shape, strides, data, free_when_done);
}



void OpusWrite(const std::string& path, const py::array_t<int16_t>& waveform_tc, const int sample_rate, const int bitrate=OPUS_AUTO, const int signal_type = 0, const int encoder_complexity = 10, const int application = 1, const int packet_loss_pct = 0) {
    if (waveform_tc.ndim() != 2) {
        throw py::value_error("waveform_tc must have exactly 2 dimension: [time, channels].");
    }
    if (waveform_tc.shape(1) > 8 || waveform_tc.shape(1) < 1) {
        throw py::value_error("waveform_tc must have at least 1 channel, and no more than 8.");
    }
    if ((bitrate < 500 or bitrate > 512000) && bitrate != OPUS_BITRATE_MAX && bitrate != OPUS_AUTO) {
        throw py::value_error("Invalid bitrate, must be at least 512 and at most 512k bits/s.");
    }
    if (sample_rate < 8000 or sample_rate > 48000) {
        throw py::value_error("Invalid sample_rate, must be at least 8k and at most 48k.");
    }
    if (encoder_complexity > 10 || encoder_complexity < 0) {
        throw py::value_error("Invalid encoder_complexity, must be in range [0, 10] inclusive. The higher, the better quality at the given bitrate, but uses more CPU.");
    }
    opus_int32 opus_signal_type;
    switch (signal_type) {
        case 0:
            opus_signal_type = OPUS_AUTO;
            break;
        case 1:
            opus_signal_type = OPUS_SIGNAL_MUSIC;
            break;
        case 2:
            opus_signal_type = OPUS_SIGNAL_VOICE;
            break;
        default:
            throw py::value_error("Invalid signal type, must be 0 (auto), 1 (music) or 2 (voice).");
    }

    opus_int32 opus_application;
    switch (application) {
        case 0:
            opus_application = OPUS_APPLICATION_VOIP;
            break;
        case 1:
            opus_application = OPUS_APPLICATION_AUDIO;
            break;
        case 2:
            opus_application = OPUS_APPLICATION_RESTRICTED_LOWDELAY;
            break;
        default:
            throw py::value_error("Invalid application type. Must be 0 (VOIP), 1 (Audio), or 2 (Restricted Low Delay).");
    }

    OggOpusComments* comments = ope_comments_create();
    //  ope_comments_add(comments, "hello", "world");
    int error;
    // We set family == 1, and channels based on waveform.
    OggOpusEnc* enc = ope_encoder_create_file(
            path.data(), comments, sample_rate, waveform_tc.shape(1), 0, &error);
    if (error != 0) {
        throw py::value_error("Unexpected error, is the provided path valid?");
    }

    if (ope_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate) != 0) {
        throw py::value_error("This should not happen. Could not set bitrate...");
    }

    // Set application type
    if (ope_encoder_ctl(enc, OPUS_SET_APPLICATION_REQUEST, opus_application) != 0) {
        throw py::value_error("This should not happen. Could not set application type...");
    }

    if (ope_encoder_ctl(enc, OPUS_SET_SIGNAL_REQUEST, opus_signal_type) != 0) {
        throw py::value_error("This should not happen. Could not set signal type...");
    }
    if (ope_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, encoder_complexity) != 0) {
        throw py::value_error("This should not happen. Could not set encoder complexity...");
    }
    if (ope_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, encoder_complexity) != 0) {
        throw py::value_error("This should not happen. Could not set packet loss percentage...");
    }

    // OK, now we are all configured. Let's write!
    if (ope_encoder_write(enc, waveform_tc.data(), waveform_tc.shape(0)) != 0) {
        throw py::value_error("Could not write audio data.");
    }
    if (ope_encoder_drain(enc) != 0) {
        throw py::value_error("Could not finalize write.");
    }

    // Cleanup.
    ope_encoder_destroy(enc);
    ope_comments_destroy(comments);
}

std::tuple<py::array_t<opus_int16>, int> OpusRead(const std::string& path, double loss_probability = 0.1) {
    int error;
    OggOpusFile* file = op_open_file(path.data(), &error);
    if (error != 0) {
        throw py::value_error("Could not open opus file.");
    }
    const ssize_t num_chans = op_channel_count(file, -1);
    const ssize_t num_samples = op_pcm_total(file, -1) / num_chans;

    const OpusHead* meta = op_head(file, -1); // unowned
    const int sample_rate = meta->input_sample_rate;

    auto* data = static_cast<opus_int16 *>(malloc(sizeof(opus_int16) * num_chans * num_samples));
    auto waveform_tc = MakeNpArray<opus_int16>({num_samples, num_chans}, data);
    size_t num_read = 0;

    // Set up a random number generator for packet loss simulation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution packet_loss_distribution(loss_probability);

    while (true) {
        // Simulate packet loss
        bool packet_lost = packet_loss_distribution(gen);
        
        int chunk;
        if (packet_lost) {
            // Simulate packet loss by calling the decoder with a null pointer
            chunk = op_read(file, nullptr, 0, nullptr);  // Lost packet
            std::cout << "Simulating packet loss (null packet)." << std::endl;

        } else {
            // Normal packet read
            chunk = op_read(file, data + num_read * num_chans, num_samples - num_read * num_chans, nullptr);
            std::cout << "Read " << chunk << " samples." << std::endl;

            if (chunk < 0) {
            throw py::value_error("Could not read opus file.");
            }
            if (chunk == 0) {
                break;
            }
        }
        num_read += chunk;
        
        if (num_read > num_samples) {
            throw py::value_error("Read too much???");
        }
    }

    if (num_read < num_samples * (1.0 - loss_probability)) {
    std::cout << "Warning: Fewer samples read than expected due to packet loss. Read: "
                  << num_read << ", Expected: " << num_samples << std::endl;
}
    op_free(file);
    return std::make_tuple(std::move(waveform_tc), sample_rate);
}


std::tuple<py::array_t<opus_int16>, int> OpusReadWithPacketLoss(const std::string& path, double loss_probability = 0.1) {
    // Open the file
    FILE* infile = fopen(path.c_str(), "rb");
    if (!infile) {
        throw py::value_error("Could not open file.");
    }

    // Initialize Ogg structures
    ogg_sync_state   oy;
    ogg_sync_init(&oy);

    ogg_page         og;
    ogg_stream_state os;

    // Initialize Opus decoder
    int error;
    OpusDecoder* decoder = opus_decoder_create(48000, 1, &error); // Set to 1 for mono
    if (error != OPUS_OK) {
        throw py::value_error("Failed to create Opus decoder.");
    }

    // Prepare for packet loss simulation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution packet_loss_distribution(loss_probability);
    std::cout << "Initializing packet loss simulation with probability: " << loss_probability << std::endl;


    // Buffer for PCM output
    std::vector<opus_int16> pcm_buffer;
    int frame_size = 960; // Expected frame size for Opus at 48 kHz with 20 ms frames
    opus_int16 pcm[frame_size]; // Mono buffer

    bool eos = false;
    while (!eos) {
        // Read data from file into the sync layer
        char* buffer = ogg_sync_buffer(&oy, 4096);
        int bytes = fread(buffer, 1, 4096, infile);
        ogg_sync_wrote(&oy, bytes);

        // Loop while there are pages
        while (ogg_sync_pageout(&oy, &og) == 1) {
            // Add page to the bitstream
            if (ogg_page_bos(&og)) {
                ogg_stream_init(&os, ogg_page_serialno(&og));
            }
            ogg_stream_pagein(&os, &og);

            // Extract packets
            ogg_packet op;
            while (ogg_stream_packetout(&os, &op) == 1) {
                bool packet_lost = packet_loss_distribution(gen);
                int decoded_samples;

                if (packet_lost) {
                    // Simulate packet loss by calling with a null pointer for concealment
                    decoded_samples = opus_decode(decoder, NULL, 0, pcm, frame_size, 0);
                    if (decoded_samples < 0) {
                        std::cerr << "Warning: Concealment error during packet loss simulation." << std::endl;
                        decoded_samples = frame_size; // Use expected frame size in case of error
                        std::fill_n(pcm, decoded_samples, 0);
                    }
                } else {
                    // Decode the packet normally
                    decoded_samples = opus_decode(decoder, (const unsigned char*)op.packet, op.bytes, pcm, frame_size, 0);
                    if (decoded_samples < 0) {
                        // Treat decoding error as packet loss, using concealment
                        std::cerr << "Warning: Decoding error, treating as packet loss." << std::endl;
                        decoded_samples = opus_decode(decoder, NULL, 0, pcm, frame_size, 0);
                        if (decoded_samples < 0) {
                            std::cerr << "Warning: Concealment error after decoding failure." << std::endl;
                            decoded_samples = frame_size;
                            std::fill_n(pcm, decoded_samples, 0);
                        }
                    }
                }

                // Append the samples (either concealed or decoded) to the buffer
                pcm_buffer.insert(pcm_buffer.end(), pcm, pcm + frame_size); // Always insert frame_size samples
            }

            if (ogg_page_eos(&og)) {
                eos = true;
            }
        }

        if (bytes == 0) {
            break; // End of file
        }
    }

    // Final cleanup and return
    ogg_stream_clear(&os);      // Clear the ogg stream state
    ogg_sync_clear(&oy);        // Clear the sync state
    fclose(infile);             // Close the file
    opus_decoder_destroy(decoder); // Destroy the decoder instance

    // Convert pcm_buffer to numpy array
    ssize_t num_samples = pcm_buffer.size(); // Total samples for mono
    auto waveform_tc = MakeNpArray<opus_int16>({num_samples, 1}, pcm_buffer.data());

    return std::make_tuple(std::move(waveform_tc), 48000);
}

//int main(int argc, char *argv[])
//{
//    int err;
//    const int sample_rate = 48000;
//    const int wave_hz = 330;
//    const opus_int16 max_ampl = std::numeric_limits<opus_int16>::max() / 2;
//    OggOpusComments* a = ope_comments_create();
//    OggOpusEnc* file = ope_encoder_create_file(
//            "hello.opus", a, sample_rate, 1, 0, &err);
//    if (ope_encoder_ctl(file, OPUS_SET_BITRATE_REQUEST, 10000) != 0) {
//        throw std::invalid_argument("Invalid bitrate.");
//    }
//
//
//    std::vector<int16_t> wave;
//    for (int i = 0; i < sample_rate*11; i++) {
//        double ampl = max_ampl * sin(static_cast<double>(i)/sample_rate*2*M_PI*wave_hz);
//        wave.push_back(static_cast<opus_int16>(ampl));
//    }
//
//    ope_encoder_write(file, wave.data(), wave.size());
//    ope_encoder_drain(file);
//    ope_encoder_destroy(file);
//}



PYBIND11_MODULE(opuspy, m) {

    m.def("write", &OpusWrite, py::arg("path"), py::arg("waveform_tc"), py::arg("sample_rate"), py::arg("bitrate")=OPUS_AUTO, py::arg("signal_type")=0, py::arg("encoder_complexity")=10, py::arg("application")=1, py::arg("packet_loss_pct")=0,
           "Saves the waveform_tc as the opus-encoded file at the specified path. The waveform must be a numpy array of np.int16 type, and shape [samples (time axis), channels]. Recommended sample rate is 48000. You can specify the bitrate in bits/s, as well as encoder_complexity (in range [0, 10] inclusive, the higher the better quality at the given bitrate, but more CPU usage, 10 is recommended). The signal_type option can help improve quality for specific audio types (0 = AUTO (default), 1 = MUSIC, 2 = SPEECH). The application parameter controls the Opus encoder mode (0 = VOIP, 1 = AUDIO, 2 = RESTRICTED_LOWDELAY).");
    m.def("read", &OpusRead, py::arg("path"), py::arg("loss_probability")=0.1, "Returns the waveform_tc as the int16 np.array of shape [samples, channels] and the original sample rate. NOTE: the waveform returned is ALWAYS at 48khz as this is how opus stores any waveform, the sample rate returned is just the original sample rate of encoded audio that you might witch to resample the returned waveform to.");
    m.def("read_with_packet_loss", &OpusReadWithPacketLoss, 
      py::arg("path"), 
      py::arg("loss_probability") = 0.1,
      "Reads the specified Opus file with simulated packet loss and returns the waveform as an int16 np.array with shape [samples, channels] and the sample rate.");
}