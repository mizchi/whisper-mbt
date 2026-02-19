///|
fn main {
  let model_path = @ffi.getenv("WHISPER_MODEL")
  let wav_path = @ffi.getenv("WHISPER_WAV")
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
  println("Loading model: " + model_path)
  let ctx = @lib.WhisperContext::init(model_path)
  match ctx {
    None => {
      println("Error: failed to load model")
      return
    }
    Some(ctx) => {
      println("Model loaded successfully")
      println("Processing: " + wav_path)
      let segments = ctx.transcribe(wav_path, language="auto")
      println("")
      println("=== Transcription ===")
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
      }
      ctx.free()
    }
  }
}
