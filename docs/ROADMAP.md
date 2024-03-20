### Features to Add:

**3. Test Decoder Dtype Influence** ✔️

* **`torch.bfloat16` vs. `torch.float16`:**
    - [x] VRAM footprint
    - [x] Inference speed
    - [x] Image quality

**2. Batch Size Fix (>1)** ✔️

* **Goal:** Restore the ability to generate multiple images per prompt.
    - [x] Not getting anywhere, opened [issue #7377](https://github.com/huggingface/diffusers/issues/7377) to hopefully get this resolved.
        - [x] **Issue Review:** Test provided solution to issue ([PR #7381](https://github.com/huggingface/diffusers/pull/7381))! Amazing work, thx [@DN6](https://github.com/DN6)! 🎉
* **Troubleshooting Steps:**
    - [ ] **Error Analysis:** Identify the specific error or unexpected behavior.
    - [ ] **Code Review:** Examine logic related to batch size handling.
    - [ ] **Dependency Check:** Ensure compatibility between any updated libraries and the batching functionality.

**1. Image Metadata Storage** ✔️

* **Goal:** Embed essential generation parameters within generated images for reproducibility and analysis.
* **Metadata to Include:**
    - [x] Seed
    - [x] Number of steps
    - [x] Model name
    - [x] CFG value
    - [x] Sampler
    - [x] Prompt

* **Implementation Steps:**
    - **Library Selection:** Research image metadata libraries (e.g., ExifWrite, PIL/Pillow).
    - **Integration:** Modify image generation code to write metadata.
    - **Testing:** Verify metadata is written and readable.
