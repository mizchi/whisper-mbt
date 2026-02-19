#include "../../include/whisper.h"
#include <moonbit.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Helper: MoonBit Bytes -> C string (NULL-terminated)
static char* bytes_to_cstring(moonbit_bytes_t bytes) {
    int32_t len = Moonbit_array_length(bytes);
    char* str = (char*)malloc(len + 1);
    memcpy(str, bytes, len);
    str[len] = '\0';
    return str;
}

// Helper: C string -> MoonBit Bytes
static moonbit_bytes_t cstring_to_bytes(const char* str) {
    if (str == NULL) {
        return moonbit_make_bytes(0, 0);
    }
    int len = (int)strlen(str);
    moonbit_bytes_t result = moonbit_make_bytes(len, 0);
    memcpy(result, str, len);
    return result;
}

// --- Context management ---

struct whisper_context* whisper_ctx_init(moonbit_bytes_t model_path) {
    char* path = bytes_to_cstring(model_path);
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(path, cparams);
    free(path);
    return ctx;
}

int32_t whisper_ctx_is_null(struct whisper_context* ctx) {
    return ctx == NULL ? 1 : 0;
}

void whisper_ctx_free(struct whisper_context* ctx) {
    if (ctx != NULL) {
        whisper_free(ctx);
    }
}

// --- Params management (heap-allocated) ---

struct whisper_full_params* whisper_params_create(void) {
    struct whisper_full_params* p = (struct whisper_full_params*)malloc(sizeof(struct whisper_full_params));
    *p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // Disable noisy output by default
    p->print_progress = false;
    p->print_realtime = false;
    p->print_special = false;
    p->print_timestamps = false;
    return p;
}

void whisper_params_set_language(struct whisper_full_params* p, moonbit_bytes_t lang) {
    // whisper expects a static string or one that outlives the call.
    // We use a static buffer since language is set once before whisper_full.
    static char lang_buffer[32];
    char* tmp = bytes_to_cstring(lang);
    strncpy(lang_buffer, tmp, sizeof(lang_buffer) - 1);
    lang_buffer[sizeof(lang_buffer) - 1] = '\0';
    free(tmp);
    p->language = lang_buffer;
}

void whisper_params_set_translate(struct whisper_full_params* p, int32_t translate) {
    p->translate = translate != 0;
}

void whisper_params_set_n_threads(struct whisper_full_params* p, int32_t n_threads) {
    p->n_threads = n_threads;
}

void whisper_params_free(struct whisper_full_params* p) {
    if (p != NULL) {
        free(p);
    }
}

// --- WAV loading (16-bit PCM mono -> float32 at 16kHz) ---

typedef struct {
    float* data;
    int count;
} wav_samples_t;

static wav_samples_t* g_last_samples = NULL;

wav_samples_t* whisper_load_wav(moonbit_bytes_t wav_path) {
    char* path = bytes_to_cstring(wav_path);
    FILE* f = fopen(path, "rb");
    free(path);
    if (!f) return NULL;

    // Read WAV header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return NULL;
    }

    uint32_t file_size;
    fread(&file_size, 4, 1, f);

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return NULL;
    }

    // Find fmt chunk
    int16_t audio_format = 0;
    int16_t num_channels = 0;
    int32_t sample_rate = 0;
    int16_t bits_per_sample = 0;

    while (1) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) { fclose(f); return NULL; }
        if (fread(&chunk_size, 4, 1, f) != 1) { fclose(f); return NULL; }

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            // skip byte_rate(4) + block_align(2)
            fseek(f, 6, SEEK_CUR);
            fread(&bits_per_sample, 2, 1, f);
            // skip remaining fmt data
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            // Found data chunk
            if (audio_format != 1 || bits_per_sample != 16) {
                // Only support PCM 16-bit
                fclose(f);
                return NULL;
            }

            int num_samples = chunk_size / (bits_per_sample / 8) / num_channels;

            // Read raw samples
            int16_t* raw = (int16_t*)malloc(num_samples * num_channels * sizeof(int16_t));
            fread(raw, sizeof(int16_t), num_samples * num_channels, f);

            // Convert to mono float32
            float* mono = (float*)malloc(num_samples * sizeof(float));
            for (int i = 0; i < num_samples; i++) {
                if (num_channels == 1) {
                    mono[i] = (float)raw[i] / 32768.0f;
                } else {
                    // Mix channels to mono
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; c++) {
                        sum += (float)raw[i * num_channels + c] / 32768.0f;
                    }
                    mono[i] = sum / num_channels;
                }
            }
            free(raw);

            // Resample to 16kHz if needed
            float* output = mono;
            int output_count = num_samples;

            if (sample_rate != 16000 && sample_rate > 0) {
                output_count = (int)((int64_t)num_samples * 16000 / sample_rate);
                output = (float*)malloc(output_count * sizeof(float));
                for (int i = 0; i < output_count; i++) {
                    float src_idx = (float)i * sample_rate / 16000.0f;
                    int idx0 = (int)src_idx;
                    float frac = src_idx - idx0;
                    if (idx0 + 1 < num_samples) {
                        output[i] = mono[idx0] * (1.0f - frac) + mono[idx0 + 1] * frac;
                    } else if (idx0 < num_samples) {
                        output[i] = mono[idx0];
                    } else {
                        output[i] = 0.0f;
                    }
                }
                free(mono);
            }

            wav_samples_t* result = (wav_samples_t*)malloc(sizeof(wav_samples_t));
            result->data = output;
            result->count = output_count;
            fclose(f);
            return result;
        } else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    fclose(f);
    return NULL;
}

int32_t whisper_samples_count(wav_samples_t* s) {
    return s ? s->count : 0;
}

int32_t whisper_samples_is_null(wav_samples_t* s) {
    return s == NULL ? 1 : 0;
}

void whisper_samples_free(wav_samples_t* s) {
    if (s) {
        if (s->data) free(s->data);
        free(s);
    }
}

// --- Inference ---

int32_t whisper_run_full(struct whisper_context* ctx, struct whisper_full_params* params, wav_samples_t* samples) {
    if (!ctx || !params || !samples || !samples->data) return -1;
    return whisper_full(ctx, *params, samples->data, samples->count);
}

int32_t whisper_run_full_parallel(struct whisper_context* ctx, struct whisper_full_params* params, wav_samples_t* samples, int32_t n_processors) {
    if (!ctx || !params || !samples || !samples->data) return -1;
    if (n_processors < 1) n_processors = 1;
    return whisper_full_parallel(ctx, *params, samples->data, samples->count, n_processors);
}

int32_t whisper_get_n_segments(struct whisper_context* ctx) {
    return whisper_full_n_segments(ctx);
}

moonbit_bytes_t whisper_get_segment_text(struct whisper_context* ctx, int32_t i) {
    const char* text = whisper_full_get_segment_text(ctx, i);
    return cstring_to_bytes(text);
}

int64_t whisper_get_segment_t0(struct whisper_context* ctx, int32_t i) {
    return whisper_full_get_segment_t0(ctx, i);
}

int64_t whisper_get_segment_t1(struct whisper_context* ctx, int32_t i) {
    return whisper_full_get_segment_t1(ctx, i);
}

// --- Group 1: Params setters ---

void whisper_params_set_offset_ms(struct whisper_full_params* p, int32_t val) {
    p->offset_ms = val;
}

void whisper_params_set_duration_ms(struct whisper_full_params* p, int32_t val) {
    p->duration_ms = val;
}

void whisper_params_set_no_timestamps(struct whisper_full_params* p, int32_t val) {
    p->no_timestamps = val != 0;
}

void whisper_params_set_single_segment(struct whisper_full_params* p, int32_t val) {
    p->single_segment = val != 0;
}

void whisper_params_set_token_timestamps(struct whisper_full_params* p, int32_t val) {
    p->token_timestamps = val != 0;
}

void whisper_params_set_max_len(struct whisper_full_params* p, int32_t val) {
    p->max_len = val;
}

void whisper_params_set_max_tokens(struct whisper_full_params* p, int32_t val) {
    p->max_tokens = val;
}

void whisper_params_set_audio_ctx(struct whisper_full_params* p, int32_t val) {
    p->audio_ctx = val;
}

void whisper_params_set_initial_prompt(struct whisper_full_params* p, moonbit_bytes_t prompt) {
    static char prompt_buffer[4096];
    char* tmp = bytes_to_cstring(prompt);
    strncpy(prompt_buffer, tmp, sizeof(prompt_buffer) - 1);
    prompt_buffer[sizeof(prompt_buffer) - 1] = '\0';
    free(tmp);
    p->initial_prompt = prompt_buffer;
}

void whisper_params_set_temperature(struct whisper_full_params* p, double val) {
    p->temperature = (float)val;
}

void whisper_params_set_print_progress(struct whisper_full_params* p, int32_t val) {
    p->print_progress = val != 0;
}

void whisper_params_set_strategy(struct whisper_full_params* p, int32_t val) {
    p->strategy = (enum whisper_sampling_strategy)val;
}

void whisper_params_set_beam_size(struct whisper_full_params* p, int32_t val) {
    p->beam_search.beam_size = val;
}

void whisper_params_set_no_context(struct whisper_full_params* p, int32_t val) {
    p->no_context = val != 0;
}

void whisper_params_set_vad(struct whisper_full_params* p, int32_t val) {
    p->vad = val != 0;
}

void whisper_params_set_vad_model_path(struct whisper_full_params* p, moonbit_bytes_t path) {
    static char vad_path_buffer[4096];
    char* tmp = bytes_to_cstring(path);
    strncpy(vad_path_buffer, tmp, sizeof(vad_path_buffer) - 1);
    vad_path_buffer[sizeof(vad_path_buffer) - 1] = '\0';
    free(tmp);
    p->vad_model_path = vad_path_buffer;
}

void whisper_params_set_vad_threshold(struct whisper_full_params* p, double val) {
    p->vad_params.threshold = (float)val;
}

void whisper_params_set_vad_min_speech_duration_ms(struct whisper_full_params* p, int32_t val) {
    p->vad_params.min_speech_duration_ms = val;
}

void whisper_params_set_vad_min_silence_duration_ms(struct whisper_full_params* p, int32_t val) {
    p->vad_params.min_silence_duration_ms = val;
}

void whisper_params_set_vad_max_speech_duration_s(struct whisper_full_params* p, double val) {
    p->vad_params.max_speech_duration_s = (float)val;
}

void whisper_params_set_vad_speech_pad_ms(struct whisper_full_params* p, int32_t val) {
    p->vad_params.speech_pad_ms = val;
}

// --- Group 2: Model info / metadata ---

int32_t whisper_ctx_is_multilingual(struct whisper_context* ctx) {
    return whisper_is_multilingual(ctx);
}

int32_t whisper_ctx_n_vocab(struct whisper_context* ctx) {
    return whisper_n_vocab(ctx);
}

int32_t whisper_ctx_n_text_ctx(struct whisper_context* ctx) {
    return whisper_n_text_ctx(ctx);
}

int32_t whisper_ctx_n_audio_ctx(struct whisper_context* ctx) {
    return whisper_n_audio_ctx(ctx);
}

moonbit_bytes_t whisper_ctx_model_type(struct whisper_context* ctx) {
    const char* s = whisper_model_type_readable(ctx);
    return cstring_to_bytes(s);
}

int32_t whisper_ctx_lang_max_id(void) {
    return whisper_lang_max_id();
}

int32_t whisper_ctx_lang_id(moonbit_bytes_t lang) {
    char* s = bytes_to_cstring(lang);
    int32_t id = whisper_lang_id(s);
    free(s);
    return id;
}

moonbit_bytes_t whisper_ctx_lang_str(int32_t id) {
    const char* s = whisper_lang_str(id);
    return cstring_to_bytes(s);
}

// --- Group 3: Segment details / token info ---

double whisper_get_segment_no_speech_prob(struct whisper_context* ctx, int32_t i) {
    return (double)whisper_full_get_segment_no_speech_prob(ctx, i);
}

int32_t whisper_get_segment_speaker_turn(struct whisper_context* ctx, int32_t i) {
    return whisper_full_get_segment_speaker_turn_next(ctx, i) ? 1 : 0;
}

int32_t whisper_get_full_lang_id(struct whisper_context* ctx) {
    return whisper_full_lang_id(ctx);
}

int32_t whisper_get_n_tokens(struct whisper_context* ctx, int32_t i_segment) {
    return whisper_full_n_tokens(ctx, i_segment);
}

moonbit_bytes_t whisper_get_token_text(struct whisper_context* ctx, int32_t i_segment, int32_t i_token) {
    const char* s = whisper_full_get_token_text(ctx, i_segment, i_token);
    return cstring_to_bytes(s);
}

int32_t whisper_get_token_id(struct whisper_context* ctx, int32_t i_segment, int32_t i_token) {
    return (int32_t)whisper_full_get_token_id(ctx, i_segment, i_token);
}

double whisper_get_token_prob(struct whisper_context* ctx, int32_t i_segment, int32_t i_token) {
    return (double)whisper_full_get_token_p(ctx, i_segment, i_token);
}

int64_t whisper_get_token_data_t0(struct whisper_context* ctx, int32_t i_segment, int32_t i_token) {
    whisper_token_data data = whisper_full_get_token_data(ctx, i_segment, i_token);
    return data.t0;
}

int64_t whisper_get_token_data_t1(struct whisper_context* ctx, int32_t i_segment, int32_t i_token) {
    whisper_token_data data = whisper_full_get_token_data(ctx, i_segment, i_token);
    return data.t1;
}

// --- Group 4: Performance / utility ---

void whisper_ctx_print_timings(struct whisper_context* ctx) {
    whisper_print_timings(ctx);
}

void whisper_ctx_reset_timings(struct whisper_context* ctx) {
    whisper_reset_timings(ctx);
}

double whisper_ctx_get_timings_sample_ms(struct whisper_context* ctx) {
    struct whisper_timings* t = whisper_get_timings(ctx);
    return t ? (double)t->sample_ms : 0.0;
}

double whisper_ctx_get_timings_encode_ms(struct whisper_context* ctx) {
    struct whisper_timings* t = whisper_get_timings(ctx);
    return t ? (double)t->encode_ms : 0.0;
}

double whisper_ctx_get_timings_decode_ms(struct whisper_context* ctx) {
    struct whisper_timings* t = whisper_get_timings(ctx);
    return t ? (double)t->decode_ms : 0.0;
}

double whisper_ctx_get_timings_batchd_ms(struct whisper_context* ctx) {
    struct whisper_timings* t = whisper_get_timings(ctx);
    return t ? (double)t->batchd_ms : 0.0;
}

double whisper_ctx_get_timings_prompt_ms(struct whisper_context* ctx) {
    struct whisper_timings* t = whisper_get_timings(ctx);
    return t ? (double)t->prompt_ms : 0.0;
}

moonbit_bytes_t whisper_get_system_info(void) {
    const char* s = whisper_print_system_info();
    return cstring_to_bytes(s);
}

// --- Environment variable access ---

moonbit_bytes_t whisper_getenv(moonbit_bytes_t name) {
    char* name_str = bytes_to_cstring(name);
    const char* val = getenv(name_str);
    free(name_str);
    return cstring_to_bytes(val);
}
