# AksharaNet Roadmap

High-accuracy Malayalam OCR engine. Target: ≥95% CER on printed text, ≥98% with full stack.

---

## Phase 1 — Build Foundation
> CMake target graph, component stubs, all libraries wired up, tests compile and pass.

- [ ] Create component directory structure (`preprocessing`, `detection`, `recognition`, `decoding`, `language_model`, `pipeline`)
- [ ] Write public header for each component (`include/aksharanet/<component>/<name>.hpp`)
- [ ] Write stub `.cpp` for each component (returns empty/trivial values)
- [ ] Write `CMakeLists.txt` for each component as a static library
- [ ] Wire CMake target graph: `aksharanet_cli → pipeline → preprocessing, detection, recognition, decoding, language_model`
- [ ] Add `aksharanet_warnings` INTERFACE target with centralised compiler flags (`-Wall -Wextra -Werror`)
- [ ] Link ICU4C (`uc`, `i18n`) into `language_model` and `decoding` targets
- [ ] Link KenLM (via FetchContent) into `language_model` target
- [ ] Link ONNX Runtime into `recognition` and `detection` targets
- [ ] Link OpenCV into `preprocessing`, `detection`, `recognition` targets
- [ ] Replace `test_stub.cpp` with per-component test files (`test_preprocessing.cpp`, `test_recognition.cpp`, etc.)
- [ ] All tests pass with `ctest --test-dir build --output-on-failure`

---

## Phase 2 — Preprocessing Pipeline
> Clean, normalised image into the recognition stage.

- [ ] Implement Sauvola adaptive binarization (handles uneven lighting, thin Malayalam strokes)
- [ ] Implement Otsu binarization as fallback
- [ ] Implement Gaussian denoising
- [ ] Implement skew detection and correction (Hough transform based)
- [ ] Implement perspective correction
- [ ] Implement contrast normalization (CLAHE)
- [ ] Preserve aspect ratio on resize — critical for Malayalam vowel signs
- [ ] Skip binarization path for clean digital renders (detect and bypass)
- [ ] Unit tests: known-skew images corrected within ±0.5°
- [ ] Unit tests: binarized output matches reference fixtures
- [ ] Benchmark: preprocessing completes in <20ms per image on CPU

---

## Phase 3 — Text Detection
> Locate text regions in full-page document images.

- [ ] Define `TextRegion` struct (bounding box, confidence score)
- [ ] Implement DBNet-based detector (ONNX model load + inference)
- [ ] Implement CRAFT-based detector as alternative
- [ ] Gate detection models behind CMake option (`AKSHARANET_ENABLE_DETECTION`)
- [ ] Post-process raw heatmaps → bounding polygons
- [ ] Non-maximum suppression on overlapping boxes
- [ ] Sort detected regions top-to-bottom, right-to-left (Malayalam reading order)
- [ ] Unit tests: detection on sample document returns ≥90% recall on ground truth boxes
- [ ] Benchmark: detection completes in <50ms per page on CPU

---

## Phase 4 — Recognition Model Integration
> Load and run the PARSeq ONNX model. Input image patch → logits.

- [ ] Define grapheme cluster vocabulary for Malayalam (~1100 entries covering full Unicode Malayalam block)
- [ ] Build vocabulary file: codepoint sequences → integer IDs
- [ ] Implement ONNX Runtime session manager (load model, manage memory, handle errors)
- [ ] Implement input tensor preparation (resize, normalize, channel order)
- [ ] Implement forward pass: image patch → raw logit tensor
- [ ] Support ViT-Base and ViT-Large encoder variants (selectable at load time)
- [ ] Gate CUDA/TensorRT execution providers behind CMake options (`AKSHARANET_ENABLE_CUDA`, `AKSHARANET_ENABLE_TENSORRT`)
- [ ] Implement CPU execution provider as default
- [ ] Unit tests: model loads without error, output tensor has expected shape
- [ ] Unit tests: known image patch produces expected top-1 grapheme cluster
- [ ] Benchmark: single patch inference <100ms on CPU

---

## Phase 5 — Beam Search Decoder
> Convert logits to grapheme cluster sequences with language model fusion.

- [ ] Implement greedy decoder (argmax baseline)
- [ ] Implement beam search decoder (configurable width, default 20)
- [ ] Implement log-probability scoring per step
- [ ] Implement KenLM shallow fusion (character 7-gram LM score added at each step)
- [ ] Handle end-of-sequence token correctly
- [ ] Handle blank/padding tokens
- [ ] Implement hypothesis deduplication in beam
- [ ] Expose beam width as runtime config parameter
- [ ] Unit tests: greedy decoder matches known reference output
- [ ] Unit tests: beam search output scores higher than greedy on held-out samples
- [ ] Unit tests: KenLM fusion improves CER on synthetic test set

---

## Phase 6 — Language Model & Unicode Post-Processing
> KenLM scoring + ICU4C normalization + chillu handling.

- [ ] Implement KenLM model loader (`.arpa` and binary format)
- [ ] Implement character n-gram query interface
- [ ] Train 7-gram character KenLM on Malayalam Wikipedia + SMC corpus
- [ ] Implement NFC Unicode normalization via ICU4C
- [ ] Implement chillu normalization (`ൻ` ↔ `ന്‍` unification)
- [ ] Implement ZWJ/ZWNJ artifact stripping from beam output
- [ ] Implement grapheme cluster segmentation using ICU4C `BreakIterator`
- [ ] Malayalam dictionary lookup for high-confidence correction
- [ ] Unit tests: NFC normalization round-trips correctly
- [ ] Unit tests: chillu forms normalise to canonical form
- [ ] Unit tests: ZWJ artifacts stripped without corrupting valid conjuncts

---

## Phase 7 — End-to-End Pipeline
> Single entry point from image to text.

- [ ] Implement `Pipeline` class orchestrating all stages in order
- [ ] Implement config struct (`PipelineConfig`) — model path, beam width, LM path, CUDA flag
- [ ] Implement single-image inference path
- [ ] Implement batch inference path
- [ ] Memory budget enforcement: abort/warn if peak usage exceeds 500MB
- [ ] Latency budget enforcement: warn if end-to-end exceeds 200ms
- [ ] Graceful error handling: missing model file, unsupported image format, empty detection
- [ ] Unit tests: end-to-end on synthetic Malayalam image returns non-empty string
- [ ] Integration test: CER ≤5% on 100-sample printed Malayalam test set
- [ ] Benchmark: full pipeline <200ms on CPU for single cropped word image

---

## Phase 8 — CLI
> Usable command-line tool.

- [ ] Argument parsing: `--image`, `--model`, `--lm`, `--beam-width`, `--output`
- [ ] Single image mode
- [ ] Batch mode (directory input)
- [ ] JSON output option (text + confidence scores per region)
- [ ] Plain text output option
- [ ] Verbose mode with per-stage timing
- [ ] `--version` flag
- [ ] Man page / `--help` text

---

## Phase 9 — Model Training (Python, separate from C++ engine)
> This phase is outside the C++ runtime but required for the `.onnx` model artifact.

- [ ] Set up Python training environment (PyTorch, HuggingFace)
- [ ] Build synthetic data generator: render Malayalam text in Rachana, AnjaliOldLipi, Meera, Noto Sans Malayalam, Manjari
- [ ] Add augmentation: noise, blur, perspective distortion, brightness variation
- [ ] Collect real annotated data (IIIT-ILST, DC Books, Wikisource)
- [ ] Define grapheme cluster tokenizer matching C++ vocabulary
- [ ] Train PARSeq model on synthetic + real data
- [ ] Fine-tune on Malayalam-specific ligature-heavy samples
- [ ] Evaluate on held-out test set: target ≥95% CER printed, ≥90% CER degraded
- [ ] Export trained model to ONNX (`torch.onnx.export`)
- [ ] Validate ONNX output matches PyTorch output on 100 samples
- [ ] Place exported model at `models/malayalam_ocr.onnx`

---

## Phase 10 — Packaging & Publish
> Build artifacts to Artifactory, versioned and reproducible.

- [ ] Add CPack configuration for binary tarball
- [ ] Version ONNX model alongside binary (semantic versioning)
- [ ] CI pipeline: build → test → package → upload to Artifactory
- [ ] Add `--build=missing` Conan profile for CI
- [ ] Tag releases on GitHub matching binary versions
- [ ] Document build reproducibility requirements

---

## Accuracy Milestones

| Milestone | CER Target | Status |
|-----------|-----------|--------|
| Stub pipeline compiles | — | ⬜ |
| End-to-end on synthetic data | ≤20% | ⬜ |
| Printed clean text | ≤5% | ⬜ |
| Printed degraded text | ≤10% | ⬜ |
| PRD target | ≤5% | ⬜ |
| Stretch goal | ≤2% | ⬜ |
