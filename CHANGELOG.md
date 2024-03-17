### 17.03.2024

* **[PR #31:](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/commit/c56d43b614f9aeb2c317910a127b140c8da92a55)**
  * Generated image filenames follow the format: `image_seed-[seed]_identifier-[UUID].png`.
  * Generation metadata (model, prompt, negative prompt, etc.) is embedded within the PNG files.

   **Example:** A generated image might be named `image_seed-12345_identifier-a8b7c6d5-4e9f-8g7h-6i5j-6k4lmn3o.png`. 


* **[PR #30:](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/commit/c56d43b614f9aeb2c317910a127b140c8da92a55)**
  * Updated torch and torchvision versions to address conflicts on Linux Mint.
  * Modified `run.py` to guarantee compatibility with Linux Mint.

   **Acknowledgement:** Thanks to @thomasmcgannon for identifying these issues in [#29](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/issues/29) !

### 16.03.2024

* **Enhanced installation process:**
   * Revised installation to enable `git pull` for updates regardless of initial installation method (ZIP download or Git clone).

* **README.md improvements:**
   * Added a table of contents for enhanced navigation.

* **Structural streamlining:**
   * Cleaned up the repo structure to remove visual clutter. 

### 15.03.2024

* **Adapted model integration:**  
    * Modified code to accommodate updates ([commit be562f9](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/commit/be562f98820d292f69870719aab556806e352fa0)) in the following model repositories:
        * stabilityai/stable-cascade 
        * stabilityai/stable-cascade-prior
