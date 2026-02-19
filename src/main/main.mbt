///|
fn main {
  let model_path = @ffi.getenv("WHISPER_MODEL")
  let wav_path = @ffi.getenv("WHISPER_WAV")
  let vad_model_path = @ffi.getenv("WHISPER_VAD_MODEL")
  if model_path == "" {
    println("Error: WHISPER_MODEL environment variable not set")
    println(
      "Usage: WHISPER_MODEL=models/ggml-base.bin WHISPER_WAV=test.wav moon run src/main --target native",
    )
    return
  }
  if wav_path == "" {
    println("Error: WHISPER_WAV environment variable not set")
    println(
      "Usage: WHISPER_MODEL=models/ggml-base.bin WHISPER_WAV=test.wav moon run src/main --target native",
    )
    return
  }
  println("System info: " + @lib.system_info())
  println("")
  println("Loading model: " + model_path)
  let ctx = @lib.WhisperContext::init(model_path)
  match ctx {
    None => {
      println("Error: failed to load model")
      return
    }
    Some(ctx) => {
      println("Model loaded successfully")
      let info = ctx.model_info()
      println(
        "Model: " +
        info.model_type +
        " | multilingual: " +
        info.is_multilingual.to_string() +
        " | vocab: " +
        info.n_vocab.to_string(),
      )
      let n_parallel = @ffi.getenv("WHISPER_PARALLEL")
      println("")
      println("Processing: " + wav_path)
      let segments = if n_parallel != "" {
        let n : Int = try {
          @strconv.from_str(n_parallel[:])
        } catch {
          _ => 2
        }
        println("Using parallel mode with " + n.to_string() + " processors")
        ctx.transcribe_parallel(
          wav_path,
          n_processors=n,
          language="auto",
          token_timestamps=true,
          vad_model_path=vad_model_path,
        )
      } else {
        ctx.transcribe(
          wav_path,
          language="auto",
          token_timestamps=true,
          vad_model_path=vad_model_path,
        )
      }
      println("")
      println("=== Transcription ===")
      let detected = ctx.detected_language()
      println("Detected language: " + detected)
      if vad_model_path != "" {
        println("VAD: enabled")
      }
      println("")
      for i = 0; i < segments.length(); i = i + 1 {
        let seg = segments[i]
        let t0_ms = seg.t0 * 10L
        let t1_ms = seg.t1 * 10L
        println(
          "[" +
          t0_ms.to_string() +
          "ms -> " +
          t1_ms.to_string() +
          "ms] " +
          seg.text,
        )
        println(
          "  no_speech_prob: " +
          seg.no_speech_prob.to_string() +
          " | speaker_turn: " +
          seg.speaker_turn_next.to_string(),
        )
        let tokens = ctx.get_tokens(i)
        let mut token_str = "  tokens:"
        for j = 0; j < tokens.length(); j = j + 1 {
          let tok = tokens[j]
          if tok.text != "" {
            token_str = token_str + "[" + tok.text + "]"
          }
        }
        println(token_str)
      }
      println("")
      ctx.print_timings()
      let timings = ctx.get_timings()
      println("")
      println("=== Timings (from API) ===")
      println("  encode: " + timings.encode_ms.to_string() + " ms")
      println("  decode: " + timings.decode_ms.to_string() + " ms")
      println("  prompt: " + timings.prompt_ms.to_string() + " ms")
      ctx.free()
    }
  }
}
