# whisper-mbt: whisper.cpp MoonBit Native Bindings

target := "native"

# Default task: check
default: check

# Format code
fmt:
    moon fmt

# Type check
check:
    moon check --deny-warn --target {{target}}

# Run tests
test:
    moon test --target {{target}}

# Update snapshot tests
test-update:
    moon test --update --target {{target}}

# Build main binary
build:
    moon build src/main --target {{target}}

# Run main
run:
    moon run src/main --target {{target}}

# Generate type definition files
info:
    moon info

# Clean build artifacts
clean:
    moon clean

# --- Setup ---

# Full setup: build whisper.cpp + download models
setup: build-whisper download-model

# Build whisper.cpp (macOS with Metal)
build-whisper:
    cmake -S vendor/whisper.cpp -B vendor/whisper.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_SERVER=OFF
    cmake --build vendor/whisper.cpp/build --config Release -j$(sysctl -n hw.logicalcpu)

# Build whisper.cpp (Linux, no GPU)
build-whisper-linux:
    cmake -S vendor/whisper.cpp -B vendor/whisper.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_SERVER=OFF
    cmake --build vendor/whisper.cpp/build --config Release -j$(nproc)

# Download ggml-base model
download-model:
    mkdir -p models
    curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Download Silero VAD model
download-vad-model:
    mkdir -p models
    curl -L -o models/ggml-silero-v5.1.2.bin https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin

# Update vendored headers from whisper.cpp build
update-headers:
    cp vendor/whisper.cpp/include/whisper.h src/ffi/include/whisper.h
    cp vendor/whisper.cpp/ggml/include/ggml.h src/ffi/include/ggml.h
    cp vendor/whisper.cpp/ggml/include/ggml-cpu.h src/ffi/include/ggml-cpu.h
    cp vendor/whisper.cpp/ggml/include/ggml-backend.h src/ffi/include/ggml-backend.h
    cp vendor/whisper.cpp/ggml/include/ggml-alloc.h src/ffi/include/ggml-alloc.h

# Pre-release check
release-check: fmt info check
