// whisper.cpp FFI declarations for MoonBit native backend

///|
type WhisperCtx

///|
type WhisperParams

///|
type WavSamples

// --- Context management ---

///|
#borrow(model_path)
extern "C" fn whisper_ctx_init(model_path : Bytes) -> WhisperCtx = "whisper_ctx_init"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_is_null(ctx : WhisperCtx) -> Int = "whisper_ctx_is_null"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_free(ctx : WhisperCtx) -> Unit = "whisper_ctx_free"

// --- Params management ---

///|
extern "C" fn whisper_params_create() -> WhisperParams = "whisper_params_create"

///|
#borrow(params, lang)
extern "C" fn whisper_params_set_language(
  params : WhisperParams,
  lang : Bytes,
) -> Unit = "whisper_params_set_language"

///|
#borrow(params)
extern "C" fn whisper_params_set_translate(
  params : WhisperParams,
  translate : Int,
) -> Unit = "whisper_params_set_translate"

///|
#borrow(params)
extern "C" fn whisper_params_set_n_threads(
  params : WhisperParams,
  n_threads : Int,
) -> Unit = "whisper_params_set_n_threads"

///|
#borrow(params)
extern "C" fn whisper_params_free(params : WhisperParams) -> Unit = "whisper_params_free"

// --- WAV loading ---

///|
#borrow(wav_path)
extern "C" fn whisper_load_wav(wav_path : Bytes) -> WavSamples = "whisper_load_wav"

///|
#borrow(samples)
extern "C" fn whisper_samples_count(samples : WavSamples) -> Int = "whisper_samples_count"

///|
#borrow(samples)
extern "C" fn whisper_samples_is_null(samples : WavSamples) -> Int = "whisper_samples_is_null"

///|
#borrow(samples)
extern "C" fn whisper_samples_free(samples : WavSamples) -> Unit = "whisper_samples_free"

// --- Inference ---

///|
#borrow(ctx, params, samples)
extern "C" fn whisper_run_full(
  ctx : WhisperCtx,
  params : WhisperParams,
  samples : WavSamples,
) -> Int = "whisper_run_full"

///|
#borrow(ctx, params, samples)
extern "C" fn whisper_run_full_parallel(
  ctx : WhisperCtx,
  params : WhisperParams,
  samples : WavSamples,
  n_processors : Int,
) -> Int = "whisper_run_full_parallel"

///|
#borrow(ctx)
extern "C" fn whisper_get_n_segments(ctx : WhisperCtx) -> Int = "whisper_get_n_segments"

///|
#borrow(ctx)
extern "C" fn whisper_get_segment_text(ctx : WhisperCtx, i : Int) -> Bytes = "whisper_get_segment_text"

///|
#borrow(ctx)
extern "C" fn whisper_get_segment_t0(ctx : WhisperCtx, i : Int) -> Int64 = "whisper_get_segment_t0"

///|
#borrow(ctx)
extern "C" fn whisper_get_segment_t1(ctx : WhisperCtx, i : Int) -> Int64 = "whisper_get_segment_t1"

// --- Group 1: Params setters ---

///|
#borrow(params)
extern "C" fn whisper_params_set_offset_ms(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_offset_ms"

///|
#borrow(params)
extern "C" fn whisper_params_set_duration_ms(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_duration_ms"

///|
#borrow(params)
extern "C" fn whisper_params_set_no_timestamps(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_no_timestamps"

///|
#borrow(params)
extern "C" fn whisper_params_set_single_segment(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_single_segment"

///|
#borrow(params)
extern "C" fn whisper_params_set_token_timestamps(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_token_timestamps"

///|
#borrow(params)
extern "C" fn whisper_params_set_max_len(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_max_len"

///|
#borrow(params)
extern "C" fn whisper_params_set_max_tokens(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_max_tokens"

///|
#borrow(params)
extern "C" fn whisper_params_set_audio_ctx(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_audio_ctx"

///|
#borrow(params, prompt)
extern "C" fn whisper_params_set_initial_prompt(
  params : WhisperParams,
  prompt : Bytes,
) -> Unit = "whisper_params_set_initial_prompt"

///|
#borrow(params)
extern "C" fn whisper_params_set_temperature(
  params : WhisperParams,
  val : Double,
) -> Unit = "whisper_params_set_temperature"

///|
#borrow(params)
extern "C" fn whisper_params_set_print_progress(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_print_progress"

///|
#borrow(params)
extern "C" fn whisper_params_set_strategy(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_strategy"

///|
#borrow(params)
extern "C" fn whisper_params_set_beam_size(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_beam_size"

///|
#borrow(params)
extern "C" fn whisper_params_set_no_context(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_no_context"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_vad"

///|
#borrow(params, path)
extern "C" fn whisper_params_set_vad_model_path(
  params : WhisperParams,
  path : Bytes,
) -> Unit = "whisper_params_set_vad_model_path"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad_threshold(
  params : WhisperParams,
  val : Double,
) -> Unit = "whisper_params_set_vad_threshold"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad_min_speech_duration_ms(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_vad_min_speech_duration_ms"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad_min_silence_duration_ms(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_vad_min_silence_duration_ms"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad_max_speech_duration_s(
  params : WhisperParams,
  val : Double,
) -> Unit = "whisper_params_set_vad_max_speech_duration_s"

///|
#borrow(params)
extern "C" fn whisper_params_set_vad_speech_pad_ms(
  params : WhisperParams,
  val : Int,
) -> Unit = "whisper_params_set_vad_speech_pad_ms"

// --- Group 2: Model info / metadata ---

///|
#borrow(ctx)
extern "C" fn whisper_ctx_is_multilingual(ctx : WhisperCtx) -> Int = "whisper_ctx_is_multilingual"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_n_vocab(ctx : WhisperCtx) -> Int = "whisper_ctx_n_vocab"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_n_text_ctx(ctx : WhisperCtx) -> Int = "whisper_ctx_n_text_ctx"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_n_audio_ctx(ctx : WhisperCtx) -> Int = "whisper_ctx_n_audio_ctx"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_model_type(ctx : WhisperCtx) -> Bytes = "whisper_ctx_model_type"

///|
extern "C" fn whisper_ctx_lang_max_id() -> Int = "whisper_ctx_lang_max_id"

///|
#borrow(lang)
extern "C" fn whisper_ctx_lang_id(lang : Bytes) -> Int = "whisper_ctx_lang_id"

///|
extern "C" fn whisper_ctx_lang_str(id : Int) -> Bytes = "whisper_ctx_lang_str"

// --- Group 3: Segment details / token info ---

///|
#borrow(ctx)
extern "C" fn whisper_get_segment_no_speech_prob(
  ctx : WhisperCtx,
  i : Int,
) -> Double = "whisper_get_segment_no_speech_prob"

///|
#borrow(ctx)
extern "C" fn whisper_get_segment_speaker_turn(
  ctx : WhisperCtx,
  i : Int,
) -> Int = "whisper_get_segment_speaker_turn"

///|
#borrow(ctx)
extern "C" fn whisper_get_full_lang_id(ctx : WhisperCtx) -> Int = "whisper_get_full_lang_id"

///|
#borrow(ctx)
extern "C" fn whisper_get_n_tokens(
  ctx : WhisperCtx,
  i_segment : Int,
) -> Int = "whisper_get_n_tokens"

///|
#borrow(ctx)
extern "C" fn whisper_get_token_text(
  ctx : WhisperCtx,
  i_segment : Int,
  i_token : Int,
) -> Bytes = "whisper_get_token_text"

///|
#borrow(ctx)
extern "C" fn whisper_get_token_id(
  ctx : WhisperCtx,
  i_segment : Int,
  i_token : Int,
) -> Int = "whisper_get_token_id"

///|
#borrow(ctx)
extern "C" fn whisper_get_token_prob(
  ctx : WhisperCtx,
  i_segment : Int,
  i_token : Int,
) -> Double = "whisper_get_token_prob"

///|
#borrow(ctx)
extern "C" fn whisper_get_token_data_t0(
  ctx : WhisperCtx,
  i_segment : Int,
  i_token : Int,
) -> Int64 = "whisper_get_token_data_t0"

///|
#borrow(ctx)
extern "C" fn whisper_get_token_data_t1(
  ctx : WhisperCtx,
  i_segment : Int,
  i_token : Int,
) -> Int64 = "whisper_get_token_data_t1"

// --- Group 4: Performance / utility ---

///|
#borrow(ctx)
extern "C" fn whisper_ctx_print_timings(ctx : WhisperCtx) -> Unit = "whisper_ctx_print_timings"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_reset_timings(ctx : WhisperCtx) -> Unit = "whisper_ctx_reset_timings"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_get_timings_sample_ms(ctx : WhisperCtx) -> Double = "whisper_ctx_get_timings_sample_ms"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_get_timings_encode_ms(ctx : WhisperCtx) -> Double = "whisper_ctx_get_timings_encode_ms"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_get_timings_decode_ms(ctx : WhisperCtx) -> Double = "whisper_ctx_get_timings_decode_ms"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_get_timings_batchd_ms(ctx : WhisperCtx) -> Double = "whisper_ctx_get_timings_batchd_ms"

///|
#borrow(ctx)
extern "C" fn whisper_ctx_get_timings_prompt_ms(ctx : WhisperCtx) -> Double = "whisper_ctx_get_timings_prompt_ms"

///|
extern "C" fn whisper_get_system_info() -> Bytes = "whisper_get_system_info"

// --- Tokenization ---

///|
#borrow(ctx, text)
extern "C" fn whisper_ctx_token_count(ctx : WhisperCtx, text : Bytes) -> Int = "whisper_ctx_token_count"

///|
#borrow(ctx, text)
extern "C" fn whisper_ctx_tokenize(
  ctx : WhisperCtx,
  text : Bytes,
  max_tokens : Int,
) -> FixedArray[Int] = "whisper_ctx_tokenize"

// --- Language detection ---

///|
extern "C" fn whisper_ctx_lang_str_full(id : Int) -> Bytes = "whisper_ctx_lang_str_full"

///|
#borrow(ctx, samples)
extern "C" fn whisper_ctx_lang_auto_detect(
  ctx : WhisperCtx,
  samples : WavSamples,
  offset_ms : Int,
  n_threads : Int,
) -> Int = "whisper_ctx_lang_auto_detect"

///|
#borrow(ctx, samples, probs_out)
extern "C" fn whisper_ctx_lang_auto_detect_with_probs(
  ctx : WhisperCtx,
  samples : WavSamples,
  offset_ms : Int,
  n_threads : Int,
  probs_out : FixedArray[Double],
) -> Int = "whisper_ctx_lang_auto_detect_with_probs"

// --- Environment ---

///|
#borrow(name)
extern "C" fn whisper_getenv(name : Bytes) -> Bytes = "whisper_getenv"

// --- Helpers ---

///|
pub fn cstring(s : String) -> Bytes {
  @encoding.encode(UTF8, s)
}

///|
pub fn bytes_to_string(b : Bytes) -> String {
  let decoder = @encoding.decoder(UTF8)
  let view : BytesView = b[:]
  decoder.decode_lossy(view)
}

///|
pub fn getenv(name : String) -> String {
  bytes_to_string(whisper_getenv(cstring(name)))
}

///|
pub fn init_context(model_path : String) -> WhisperCtx? {
  let ctx = whisper_ctx_init(cstring(model_path))
  if whisper_ctx_is_null(ctx) == 1 {
    None
  } else {
    Some(ctx)
  }
}

///|
pub fn free_context(ctx : WhisperCtx) -> Unit {
  whisper_ctx_free(ctx)
}

///|
pub fn create_params() -> WhisperParams {
  whisper_params_create()
}

///|
pub fn set_language(params : WhisperParams, lang : String) -> Unit {
  whisper_params_set_language(params, cstring(lang))
}

///|
pub fn set_translate(params : WhisperParams, translate : Bool) -> Unit {
  whisper_params_set_translate(params, if translate { 1 } else { 0 })
}

///|
pub fn set_n_threads(params : WhisperParams, n : Int) -> Unit {
  whisper_params_set_n_threads(params, n)
}

///|
pub fn free_params(params : WhisperParams) -> Unit {
  whisper_params_free(params)
}

///|
pub fn load_wav(wav_path : String) -> WavSamples? {
  let samples = whisper_load_wav(cstring(wav_path))
  if whisper_samples_is_null(samples) == 1 {
    None
  } else {
    Some(samples)
  }
}

///|
pub fn samples_count(samples : WavSamples) -> Int {
  whisper_samples_count(samples)
}

///|
pub fn free_samples(samples : WavSamples) -> Unit {
  whisper_samples_free(samples)
}

///|
pub fn run_full(
  ctx : WhisperCtx,
  params : WhisperParams,
  samples : WavSamples,
) -> Int {
  whisper_run_full(ctx, params, samples)
}

///|
pub fn get_n_segments(ctx : WhisperCtx) -> Int {
  whisper_get_n_segments(ctx)
}

///|
pub fn get_segment_text(ctx : WhisperCtx, i : Int) -> String {
  bytes_to_string(whisper_get_segment_text(ctx, i))
}

///|
pub fn get_segment_t0(ctx : WhisperCtx, i : Int) -> Int64 {
  whisper_get_segment_t0(ctx, i)
}

///|
pub fn get_segment_t1(ctx : WhisperCtx, i : Int) -> Int64 {
  whisper_get_segment_t1(ctx, i)
}

// --- Group 1: Params setters (pub) ---

///|
pub fn set_offset_ms(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_offset_ms(params, val)
}

///|
pub fn set_duration_ms(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_duration_ms(params, val)
}

///|
pub fn set_no_timestamps(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_no_timestamps(params, if val { 1 } else { 0 })
}

///|
pub fn set_single_segment(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_single_segment(params, if val { 1 } else { 0 })
}

///|
pub fn set_token_timestamps(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_token_timestamps(params, if val { 1 } else { 0 })
}

///|
pub fn set_max_len(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_max_len(params, val)
}

///|
pub fn set_max_tokens(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_max_tokens(params, val)
}

///|
pub fn set_audio_ctx(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_audio_ctx(params, val)
}

///|
pub fn set_initial_prompt(params : WhisperParams, prompt : String) -> Unit {
  whisper_params_set_initial_prompt(params, cstring(prompt))
}

///|
pub fn set_temperature(params : WhisperParams, val : Double) -> Unit {
  whisper_params_set_temperature(params, val)
}

///|
pub fn set_print_progress(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_print_progress(params, if val { 1 } else { 0 })
}

///|
pub fn set_strategy(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_strategy(params, val)
}

///|
pub fn set_beam_size(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_beam_size(params, val)
}

///|
pub fn set_no_context(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_no_context(params, if val { 1 } else { 0 })
}

///|
pub fn set_vad(params : WhisperParams, val : Bool) -> Unit {
  whisper_params_set_vad(params, if val { 1 } else { 0 })
}

///|
pub fn set_vad_model_path(params : WhisperParams, path : String) -> Unit {
  whisper_params_set_vad_model_path(params, cstring(path))
}

///|
pub fn set_vad_threshold(params : WhisperParams, val : Double) -> Unit {
  whisper_params_set_vad_threshold(params, val)
}

///|
pub fn set_vad_min_speech_duration_ms(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_vad_min_speech_duration_ms(params, val)
}

///|
pub fn set_vad_min_silence_duration_ms(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_vad_min_silence_duration_ms(params, val)
}

///|
pub fn set_vad_max_speech_duration_s(params : WhisperParams, val : Double) -> Unit {
  whisper_params_set_vad_max_speech_duration_s(params, val)
}

///|
pub fn set_vad_speech_pad_ms(params : WhisperParams, val : Int) -> Unit {
  whisper_params_set_vad_speech_pad_ms(params, val)
}

///|
pub fn run_full_parallel(
  ctx : WhisperCtx,
  params : WhisperParams,
  samples : WavSamples,
  n_processors : Int,
) -> Int {
  whisper_run_full_parallel(ctx, params, samples, n_processors)
}

// --- Group 2: Model info (pub) ---

///|
pub fn is_multilingual(ctx : WhisperCtx) -> Bool {
  whisper_ctx_is_multilingual(ctx) != 0
}

///|
pub fn n_vocab(ctx : WhisperCtx) -> Int {
  whisper_ctx_n_vocab(ctx)
}

///|
pub fn n_text_ctx(ctx : WhisperCtx) -> Int {
  whisper_ctx_n_text_ctx(ctx)
}

///|
pub fn n_audio_ctx(ctx : WhisperCtx) -> Int {
  whisper_ctx_n_audio_ctx(ctx)
}

///|
pub fn model_type(ctx : WhisperCtx) -> String {
  bytes_to_string(whisper_ctx_model_type(ctx))
}

///|
pub fn lang_max_id() -> Int {
  whisper_ctx_lang_max_id()
}

///|
pub fn lang_id(lang : String) -> Int {
  whisper_ctx_lang_id(cstring(lang))
}

///|
pub fn lang_str(id : Int) -> String {
  bytes_to_string(whisper_ctx_lang_str(id))
}

// --- Group 3: Segment details / token info (pub) ---

///|
pub fn get_segment_no_speech_prob(ctx : WhisperCtx, i : Int) -> Double {
  whisper_get_segment_no_speech_prob(ctx, i)
}

///|
pub fn get_segment_speaker_turn_next(ctx : WhisperCtx, i : Int) -> Bool {
  whisper_get_segment_speaker_turn(ctx, i) != 0
}

///|
pub fn get_full_lang_id(ctx : WhisperCtx) -> Int {
  whisper_get_full_lang_id(ctx)
}

///|
pub fn get_n_tokens(ctx : WhisperCtx, i_segment : Int) -> Int {
  whisper_get_n_tokens(ctx, i_segment)
}

///|
pub fn get_token_text(ctx : WhisperCtx, i_segment : Int, i_token : Int) -> String {
  bytes_to_string(whisper_get_token_text(ctx, i_segment, i_token))
}

///|
pub fn get_token_id(ctx : WhisperCtx, i_segment : Int, i_token : Int) -> Int {
  whisper_get_token_id(ctx, i_segment, i_token)
}

///|
pub fn get_token_prob(ctx : WhisperCtx, i_segment : Int, i_token : Int) -> Double {
  whisper_get_token_prob(ctx, i_segment, i_token)
}

///|
pub fn get_token_data_t0(ctx : WhisperCtx, i_segment : Int, i_token : Int) -> Int64 {
  whisper_get_token_data_t0(ctx, i_segment, i_token)
}

///|
pub fn get_token_data_t1(ctx : WhisperCtx, i_segment : Int, i_token : Int) -> Int64 {
  whisper_get_token_data_t1(ctx, i_segment, i_token)
}

// --- Group 4: Performance / utility (pub) ---

///|
pub fn print_timings(ctx : WhisperCtx) -> Unit {
  whisper_ctx_print_timings(ctx)
}

///|
pub fn reset_timings(ctx : WhisperCtx) -> Unit {
  whisper_ctx_reset_timings(ctx)
}

///|
pub fn get_timings_sample_ms(ctx : WhisperCtx) -> Double {
  whisper_ctx_get_timings_sample_ms(ctx)
}

///|
pub fn get_timings_encode_ms(ctx : WhisperCtx) -> Double {
  whisper_ctx_get_timings_encode_ms(ctx)
}

///|
pub fn get_timings_decode_ms(ctx : WhisperCtx) -> Double {
  whisper_ctx_get_timings_decode_ms(ctx)
}

///|
pub fn get_timings_batchd_ms(ctx : WhisperCtx) -> Double {
  whisper_ctx_get_timings_batchd_ms(ctx)
}

///|
pub fn get_timings_prompt_ms(ctx : WhisperCtx) -> Double {
  whisper_ctx_get_timings_prompt_ms(ctx)
}

///|
pub fn system_info() -> String {
  bytes_to_string(whisper_get_system_info())
}

// --- Tokenization (pub) ---

///|
pub fn token_count(ctx : WhisperCtx, text : String) -> Int {
  whisper_ctx_token_count(ctx, cstring(text))
}

///|
pub fn tokenize(ctx : WhisperCtx, text : String, max_tokens : Int) -> FixedArray[Int] {
  whisper_ctx_tokenize(ctx, cstring(text), max_tokens)
}

// --- Language detection (pub) ---

///|
pub fn lang_str_full(id : Int) -> String {
  bytes_to_string(whisper_ctx_lang_str_full(id))
}

///|
pub fn lang_auto_detect(
  ctx : WhisperCtx,
  samples : WavSamples,
  offset_ms : Int,
  n_threads : Int,
) -> Int {
  whisper_ctx_lang_auto_detect(ctx, samples, offset_ms, n_threads)
}

///|
pub fn lang_auto_detect_with_probs(
  ctx : WhisperCtx,
  samples : WavSamples,
  offset_ms : Int,
  n_threads : Int,
  probs_out : FixedArray[Double],
) -> Int {
  whisper_ctx_lang_auto_detect_with_probs(ctx, samples, offset_ms, n_threads, probs_out)
}
