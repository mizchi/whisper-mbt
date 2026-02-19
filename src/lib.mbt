///|
pub struct Segment {
  text : String
  t0 : Int64
  t1 : Int64
  no_speech_prob : Double
  speaker_turn_next : Bool
} derive(Show)

///|
pub struct TokenData {
  text : String
  id : Int
  prob : Double
  t0 : Int64
  t1 : Int64
} derive(Show)

///|
pub struct ModelInfo {
  model_type : String
  is_multilingual : Bool
  n_vocab : Int
  n_text_ctx : Int
  n_audio_ctx : Int
} derive(Show)

///|
pub struct Timings {
  sample_ms : Double
  encode_ms : Double
  decode_ms : Double
  batchd_ms : Double
  prompt_ms : Double
} derive(Show)

///| Sampling strategy for whisper decoding.
/// `Greedy` (default) or `BeamSearch` with configurable beam size.
pub(all) enum Strategy {
  Greedy
  BeamSearch
} derive(Show)

///|
pub struct VadParams {
  threshold : Double
  min_speech_duration_ms : Int
  min_silence_duration_ms : Int
  max_speech_duration_s : Double
  speech_pad_ms : Int
} derive(Show)

///|
pub fn VadParams::default() -> VadParams {
  {
    threshold: 0.5,
    min_speech_duration_ms: 250,
    min_silence_duration_ms: 100,
    max_speech_duration_s: 30.0,
    speech_pad_ms: 30,
  }
}

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
fn apply_params(
  params : @ffi.WhisperParams,
  language : String,
  translate : Bool,
  n_threads : Int,
  offset_ms : Int,
  duration_ms : Int,
  no_timestamps : Bool,
  single_segment : Bool,
  token_timestamps : Bool,
  max_len : Int,
  max_tokens : Int,
  audio_ctx : Int,
  initial_prompt : String,
  temperature : Double,
  print_progress : Bool,
  strategy : Strategy,
  beam_size : Int,
  no_context : Bool,
  vad_model_path : String,
  vad_params : VadParams?,
) -> Unit {
  @ffi.set_language(params, language)
  @ffi.set_translate(params, translate)
  @ffi.set_n_threads(params, n_threads)
  if offset_ms != 0 {
    @ffi.set_offset_ms(params, offset_ms)
  }
  if duration_ms != 0 {
    @ffi.set_duration_ms(params, duration_ms)
  }
  if no_timestamps {
    @ffi.set_no_timestamps(params, true)
  }
  if single_segment {
    @ffi.set_single_segment(params, true)
  }
  if token_timestamps {
    @ffi.set_token_timestamps(params, true)
  }
  if max_len != 0 {
    @ffi.set_max_len(params, max_len)
  }
  if max_tokens != 0 {
    @ffi.set_max_tokens(params, max_tokens)
  }
  if audio_ctx != 0 {
    @ffi.set_audio_ctx(params, audio_ctx)
  }
  if initial_prompt != "" {
    @ffi.set_initial_prompt(params, initial_prompt)
  }
  if temperature != 0.0 {
    @ffi.set_temperature(params, temperature)
  }
  if print_progress {
    @ffi.set_print_progress(params, true)
  }
  match strategy {
    BeamSearch => {
      @ffi.set_strategy(params, 1)
      @ffi.set_beam_size(params, beam_size)
    }
    Greedy => ()
  }
  if no_context {
    @ffi.set_no_context(params, true)
  }
  if vad_model_path != "" {
    @ffi.set_vad(params, true)
    @ffi.set_vad_model_path(params, vad_model_path)
    match vad_params {
      Some(vp) => {
        @ffi.set_vad_threshold(params, vp.threshold)
        @ffi.set_vad_min_speech_duration_ms(params, vp.min_speech_duration_ms)
        @ffi.set_vad_min_silence_duration_ms(
          params, vp.min_silence_duration_ms,
        )
        @ffi.set_vad_max_speech_duration_s(params, vp.max_speech_duration_s)
        @ffi.set_vad_speech_pad_ms(params, vp.speech_pad_ms)
      }
      None => ()
    }
  }
}

///|
fn WhisperContext::collect_segments(self : WhisperContext) -> Array[Segment] {
  let segments : Array[Segment] = []
  let n = @ffi.get_n_segments(self.handle)
  for i = 0; i < n; i = i + 1 {
    segments.push({
      text: @ffi.get_segment_text(self.handle, i),
      t0: @ffi.get_segment_t0(self.handle, i),
      t1: @ffi.get_segment_t1(self.handle, i),
      no_speech_prob: @ffi.get_segment_no_speech_prob(self.handle, i),
      speaker_turn_next: @ffi.get_segment_speaker_turn_next(self.handle, i),
    })
  }
  segments
}

///|
pub fn WhisperContext::transcribe(
  self : WhisperContext,
  wav_path : String,
  language? : String = "en",
  translate? : Bool = false,
  n_threads? : Int = 4,
  offset_ms? : Int = 0,
  duration_ms? : Int = 0,
  no_timestamps? : Bool = false,
  single_segment? : Bool = false,
  token_timestamps? : Bool = false,
  max_len? : Int = 0,
  max_tokens? : Int = 0,
  audio_ctx? : Int = 0,
  initial_prompt? : String = "",
  temperature? : Double = 0.0,
  print_progress? : Bool = false,
  strategy? : Strategy = Greedy,
  beam_size? : Int = 5,
  no_context? : Bool = false,
  vad_model_path? : String = "",
  vad_params? : VadParams? = None,
) -> Array[Segment] {
  let params = @ffi.create_params()
  apply_params(
    params,
    language,
    translate,
    n_threads,
    offset_ms,
    duration_ms,
    no_timestamps,
    single_segment,
    token_timestamps,
    max_len,
    max_tokens,
    audio_ctx,
    initial_prompt,
    temperature,
    print_progress,
    strategy,
    beam_size,
    no_context,
    vad_model_path,
    vad_params,
  )
  let samples = @ffi.load_wav(wav_path)
  match samples {
    None => {
      @ffi.free_params(params)
      println("Error: failed to load WAV file: " + wav_path)
      return []
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
      @ffi.free_samples(s)
      @ffi.free_params(params)
      if rc != 0 {
        println("Error: whisper_full returned " + rc.to_string())
        return []
      }
      self.collect_segments()
    }
  }
}

///|
pub fn WhisperContext::transcribe_parallel(
  self : WhisperContext,
  wav_path : String,
  n_processors? : Int = 4,
  language? : String = "en",
  translate? : Bool = false,
  n_threads? : Int = 4,
  offset_ms? : Int = 0,
  duration_ms? : Int = 0,
  no_timestamps? : Bool = false,
  single_segment? : Bool = false,
  token_timestamps? : Bool = false,
  max_len? : Int = 0,
  max_tokens? : Int = 0,
  audio_ctx? : Int = 0,
  initial_prompt? : String = "",
  temperature? : Double = 0.0,
  print_progress? : Bool = false,
  strategy? : Strategy = Greedy,
  beam_size? : Int = 5,
  no_context? : Bool = false,
  vad_model_path? : String = "",
  vad_params? : VadParams? = None,
) -> Array[Segment] {
  let params = @ffi.create_params()
  apply_params(
    params,
    language,
    translate,
    n_threads,
    offset_ms,
    duration_ms,
    no_timestamps,
    single_segment,
    token_timestamps,
    max_len,
    max_tokens,
    audio_ctx,
    initial_prompt,
    temperature,
    print_progress,
    strategy,
    beam_size,
    no_context,
    vad_model_path,
    vad_params,
  )
  let samples = @ffi.load_wav(wav_path)
  match samples {
    None => {
      @ffi.free_params(params)
      println("Error: failed to load WAV file: " + wav_path)
      return []
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
      let rc = @ffi.run_full_parallel(self.handle, params, s, n_processors)
      @ffi.free_samples(s)
      @ffi.free_params(params)
      if rc != 0 {
        println("Error: whisper_full_parallel returned " + rc.to_string())
        return []
      }
      self.collect_segments()
    }
  }
}

///|
pub fn WhisperContext::get_tokens(
  self : WhisperContext,
  segment_index : Int,
) -> Array[TokenData] {
  let n = @ffi.get_n_tokens(self.handle, segment_index)
  let tokens : Array[TokenData] = []
  for i = 0; i < n; i = i + 1 {
    tokens.push({
      text: @ffi.get_token_text(self.handle, segment_index, i),
      id: @ffi.get_token_id(self.handle, segment_index, i),
      prob: @ffi.get_token_prob(self.handle, segment_index, i),
      t0: @ffi.get_token_data_t0(self.handle, segment_index, i),
      t1: @ffi.get_token_data_t1(self.handle, segment_index, i),
    })
  }
  tokens
}

///|
pub fn WhisperContext::model_info(self : WhisperContext) -> ModelInfo {
  {
    model_type: @ffi.model_type(self.handle),
    is_multilingual: @ffi.is_multilingual(self.handle),
    n_vocab: @ffi.n_vocab(self.handle),
    n_text_ctx: @ffi.n_text_ctx(self.handle),
    n_audio_ctx: @ffi.n_audio_ctx(self.handle),
  }
}

///|
pub fn WhisperContext::detected_language(self : WhisperContext) -> String {
  let lang_id = @ffi.get_full_lang_id(self.handle)
  @ffi.lang_str(lang_id)
}

///|
pub fn WhisperContext::print_timings(self : WhisperContext) -> Unit {
  @ffi.print_timings(self.handle)
}

///|
pub fn WhisperContext::reset_timings(self : WhisperContext) -> Unit {
  @ffi.reset_timings(self.handle)
}

///|
pub fn WhisperContext::get_timings(self : WhisperContext) -> Timings {
  {
    sample_ms: @ffi.get_timings_sample_ms(self.handle),
    encode_ms: @ffi.get_timings_encode_ms(self.handle),
    decode_ms: @ffi.get_timings_decode_ms(self.handle),
    batchd_ms: @ffi.get_timings_batchd_ms(self.handle),
    prompt_ms: @ffi.get_timings_prompt_ms(self.handle),
  }
}

///|
pub fn WhisperContext::free(self : WhisperContext) -> Unit {
  @ffi.free_context(self.handle)
}

///|
pub fn lang_max_id() -> Int {
  @ffi.lang_max_id()
}

///|
pub fn lang_id(lang : String) -> Int {
  @ffi.lang_id(lang)
}

///|
pub fn lang_str(id : Int) -> String {
  @ffi.lang_str(id)
}

///|
pub fn system_info() -> String {
  @ffi.system_info()
}
