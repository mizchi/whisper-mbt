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
