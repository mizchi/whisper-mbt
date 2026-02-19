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

# Run main
run:
    moon run src/main --target {{target}}

# Generate type definition files
info:
    moon info

# Clean build artifacts
clean:
    moon clean

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
    mkdir -p include
    ln -sf ../vendor/whisper.cpp/include/whisper.h include/whisper.h
    for f in vendor/whisper.cpp/ggml/include/*.h; do ln -sf "../$$f" "include/$$(basename $$f)"; done

# Download ggml-base model
download-model:
    mkdir -p models
    curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Pre-release check
release-check: fmt info check
