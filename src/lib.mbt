///|
pub struct Segment {
  text : String
  t0 : Int64
  t1 : Int64
} derive(Show)

///|
pub struct WhisperContext {
  priv handle : @ffi.WhisperCtx
}

///|
pub fn WhisperContext::init(model_path : String) -> WhisperContext? {
  match @ffi.init_context(model_path) {
    Some(ctx) => Some({ handle: ctx })
    None => None
  }
}

///|
pub fn WhisperContext::transcribe(
  self : WhisperContext,
  wav_path : String,
  language? : String = "en",
  translate? : Bool = false,
  n_threads? : Int = 4,
) -> Array[Segment] {
  let params = @ffi.create_params()
  @ffi.set_language(params, language)
  @ffi.set_translate(params, translate)
  @ffi.set_n_threads(params, n_threads)
  let samples = @ffi.load_wav(wav_path)
  let segments : Array[Segment] = []
  match samples {
    None => {
      @ffi.free_params(params)
      println("Error: failed to load WAV file: " + wav_path)
      return segments
    }
    Some(s) => {
      let n_samples = @ffi.samples_count(s)
      println(
        "Loaded " +
        n_samples.to_string() +
        " samples (" +
        (n_samples / 16000).to_string() +
        "s)",
      )
      let rc = @ffi.run_full(self.handle, params, s)
      if rc == 0 {
        let n = @ffi.get_n_segments(self.handle)
        for i = 0; i < n; i = i + 1 {
          segments.push({
            text: @ffi.get_segment_text(self.handle, i),
            t0: @ffi.get_segment_t0(self.handle, i),
            t1: @ffi.get_segment_t1(self.handle, i),
          })
        }
      } else {
        println("Error: whisper_full returned " + rc.to_string())
      }
      @ffi.free_samples(s)
    }
  }
  @ffi.free_params(params)
  segments
}

///|
pub fn WhisperContext::free(self : WhisperContext) -> Unit {
  @ffi.free_context(self.handle)
}
