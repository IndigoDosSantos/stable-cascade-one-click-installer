### Features to Add:

**1. Image Metadata Storage**

* **Goal:** Embed essential generation parameters within generated images for reproducibility and analysis.
* **Metadata to Include:**
    - [x] Seed
    - [x] Number of steps
    - [x] Model name
    - [x] CFG value
    - [ ] Sampler
    - [x] Prompt

* **Implementation Steps:**
    - **Library Selection:** Research image metadata libraries (e.g., ExifWrite, PIL/Pillow).
    - **Integration:** Modify image generation code to write metadata.
    - **Testing:** Verify metadata is written and readable.

**2. Batch Size Fix (>1)**

* **Goal:** Restore the ability to generate multiple images per prompt.
* **Troubleshooting Steps:**
    - [ ] **Error Analysis:** Identify the specific error or unexpected behavior.
    - [ ] **Code Review:** Examine logic related to batch size handling.
    - [ ] **Dependency Check:** Ensure compatibility between any updated libraries and the batching functionality.
