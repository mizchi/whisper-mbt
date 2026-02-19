# whisper

MoonBit native bindings for [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — speech-to-text inference.

## Prerequisites

- [MoonBit toolchain](https://www.moonbitlang.com/download)
- CMake, C/C++ compiler
- (macOS) Xcode Command Line Tools

## Quick Start (standalone)

```bash
git clone --recursive https://github.com/mizchi/whisper.git
cd whisper

# Build whisper.cpp and download model
just setup

# Run
WHISPER_MODEL=models/ggml-base.bin \
WHISPER_WAV=vendor/whisper.cpp/samples/jfk.wav \
  moon run src/main --target native
```

## Use as a dependency

### 1. Add the package

From mooncakes registry (when published):

```bash
moon add mizchi/whisper
```

Or as a local path dependency in `moon.mod.json`:

```json
{
  "name": "your/project",
  "deps": {
    "mizchi/whisper": {
      "path": "/path/to/whisper"
    }
  },
  "preferred-target": "native"
}
```

### 2. Build whisper.cpp

Build whisper.cpp somewhere on your system:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

# macOS (Metal)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)

# Linux (CPU only)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF
cmake --build build --config Release -j$(nproc)
```

### 3. Configure your main package

In your main package's `moon.pkg`, import the library and add link flags pointing to the whisper.cpp build:

```
// cmd/main/moon.pkg
import {
  "mizchi/whisper" @whisper,
}

options(
  "is-main": true,
  link: {
    "native": {
      // macOS (Metal + Accelerate)
      "cc-link-flags": "/path/to/whisper.cpp/build/src/libwhisper.a /path/to/whisper.cpp/build/ggml/src/libggml.a /path/to/whisper.cpp/build/ggml/src/libggml-base.a /path/to/whisper.cpp/build/ggml/src/libggml-cpu.a /path/to/whisper.cpp/build/ggml/src/ggml-metal/libggml-metal.a /path/to/whisper.cpp/build/ggml/src/ggml-blas/libggml-blas.a -lstdc++ -framework Accelerate -framework Metal -framework Foundation -framework MetalKit",
    },
  },
  "supported-targets": [ "native" ],
)
```

For Linux (CPU only):

```
"cc-link-flags": "/path/to/whisper.cpp/build/src/libwhisper.a /path/to/whisper.cpp/build/ggml/src/libggml.a /path/to/whisper.cpp/build/ggml/src/libggml-base.a /path/to/whisper.cpp/build/ggml/src/libggml-cpu.a -lstdc++ -lm -lpthread"
```

Note: `cc-link-flags` requires absolute paths or paths relative to the project root where `moon build` is invoked.

### 4. Download a model

```bash
# ggml-base (147 MB)
curl -L -o models/ggml-base.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

See [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp) for other sizes (tiny, small, medium, large).

### 5. Use

```moonbit
fn main {
  let ctx = @whisper.WhisperContext::init("models/ggml-base.bin")
  match ctx {
    None => println("Failed to load model")
    Some(ctx) => {
      let segments = ctx.transcribe("audio.wav", language="auto")
      for i = 0; i < segments.length(); i = i + 1 {
        println(segments[i].text)
      }
      ctx.free()
    }
  }
}
```

## API

### `WhisperContext`

```moonbit
WhisperContext::init(model_path : String) -> WhisperContext?
WhisperContext::transcribe(self, wav_path, language?="en", translate?=false, n_threads?=4, ...) -> Array[Segment]
WhisperContext::transcribe_parallel(self, wav_path, n_processors?=4, ...) -> Array[Segment]
WhisperContext::get_tokens(self, segment_index) -> Array[TokenData]
WhisperContext::model_info(self) -> ModelInfo
WhisperContext::detected_language(self) -> String
WhisperContext::get_timings(self) -> Timings
WhisperContext::print_timings(self) -> Unit
WhisperContext::free(self) -> Unit
```

### `transcribe` options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `String` | `"en"` | Language code or `"auto"` |
| `translate` | `Bool` | `false` | Translate to English |
| `n_threads` | `Int` | `4` | Number of threads |
| `offset_ms` | `Int` | `0` | Start offset in ms |
| `duration_ms` | `Int` | `0` | Duration to process (0 = all) |
| `no_timestamps` | `Bool` | `false` | Disable timestamps |
| `single_segment` | `Bool` | `false` | Force single segment |
| `token_timestamps` | `Bool` | `false` | Token-level timestamps |
| `max_len` | `Int` | `0` | Max segment length (chars) |
| `max_tokens` | `Int` | `0` | Max tokens per segment |
| `initial_prompt` | `String` | `""` | Initial prompt |
| `temperature` | `Double` | `0.0` | Decoding temperature |
| `strategy` | `Strategy` | `Greedy` | `Greedy` or `BeamSearch` |
| `beam_size` | `Int` | `5` | Beam size (when BeamSearch) |
| `no_context` | `Bool` | `false` | Disable past context |
| `vad_model_path` | `String` | `""` | Path to Silero VAD model (enables VAD) |
| `vad_params` | `VadParams?` | `None` | VAD tuning parameters |

### VAD (Voice Activity Detection)

VAD skips silence and processes only speech segments. Requires a separate Silero VAD model:

```bash
curl -L -o models/ggml-silero-v5.1.2.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

```moonbit
let segments = ctx.transcribe(
  "audio.wav",
  language="auto",
  vad_model_path="models/ggml-silero-v5.1.2.bin",
)
```

### Parallel inference

Splits audio across multiple processors for faster throughput:

```moonbit
let segments = ctx.transcribe_parallel(
  "long_audio.wav",
  n_processors=4,
  language="auto",
)
```

## Updating vendored headers

When upgrading the whisper.cpp submodule:

```bash
just build-whisper
just update-headers
```

## License

MIT
