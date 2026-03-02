# AksharaNet

Malayalam OCR that actually works.

A high-accuracy Malayalam OCR engine built in C++. Uses a Vision Transformer encoder with a Transformer decoder for recognition, ONNX Runtime for inference, and KenLM for language correction. Targets ≥95% CER on printed text.

---

## Architecture

```
Image → Preprocessing → Detection → Recognition → Decoding → Language Model → Text
                                          ↑
                                 models/malayalam_ocr.onnx
```

| Stage | What it does |
|-------|-------------|
| **Preprocessing** | Sauvola binarization, denoising, skew/perspective correction |
| **Detection** | Optional full-page text region detection (DBNet/CRAFT) |
| **Recognition** | ViT encoder + Transformer decoder via ONNX Runtime |
| **Decoding** | Beam search (width 20) over Malayalam grapheme cluster vocabulary |
| **Language Model** | 7-gram KenLM + ICU4C Unicode normalization + chillu correction |

---

## Building

**Requirements:** CMake ≥ 3.24, Conan 2.x, C++20 compiler

```bash
# Install dependencies
conan install . --build=missing -s build_type=Release

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake

# Build
cmake --build build -j$(nproc)
```

### Debug build

```bash
conan install . --build=missing -s build_type=Debug
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_TOOLCHAIN_FILE=build-debug/Debug/generators/conan_toolchain.cmake
cmake --build build-debug -j$(nproc)
```

### Optional CMake flags

| Flag | Default | Description |
|------|---------|-------------|
| `AKSHARANET_ENABLE_CUDA` | OFF | Enable ONNX Runtime CUDA execution provider |
| `AKSHARANET_ENABLE_TENSORRT` | OFF | Enable TensorRT execution provider |
| `AKSHARANET_ENABLE_DETECTION` | OFF | Build text detection stage |
| `AKSHARANET_SYSTEM_KENLM` | OFF | Use system KenLM instead of FetchContent |

---

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run a specific component
./build/tests/test_preprocessing

# Run a specific test case
./build/tests/test_decoding --gtest_filter=Decoding.DecodeEmptyLogitsReturnsEmptyString
```

---

## Usage

```bash
./build/src/aksharanet_cli <image_path>
```

---

## Project Structure

```
src/
├── preprocessing/     Sauvola binarization, skew correction
├── detection/         Text region detection (DBNet/CRAFT)
├── recognition/       ONNX Runtime inference wrapper
├── decoding/          Beam search decoder
├── language_model/    KenLM + ICU4C post-processing
└── pipeline/          End-to-end orchestration

models/                ONNX model artifacts (not checked in)
tests/                 GoogleTest per-component test suites
cmake/                 FetchContent modules (KenLM)
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.9.0 | Image preprocessing |
| ONNX Runtime | 1.18.1 | Model inference |
| ICU4C | 73.2 | Unicode normalization, grapheme clusters |
| KenLM | HEAD | N-gram language model |
| GoogleTest | 1.14.0 | Testing |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full implementation plan.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| CER (printed clean) | ≤ 5% |
| Latency (CPU, single image) | < 200ms |
| Memory (CPU) | < 500MB |
