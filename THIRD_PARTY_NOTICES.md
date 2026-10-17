# Third-party components

This submission uses these unmodified model checkpoints from Qwen:

- Qwen/Qwen3-VL-8B-Instruct, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen/Qwen3-Embedding-0.6B, revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

Both repositories identify their model license as Apache License 2.0. A copy of
that license is included in `licenses/APACHE-2.0.txt`. Original model README and
license files, when present, are included with the downloaded checkpoints.

Python dependencies retain the license files and package metadata supplied in
their distributions. The mandated AMD ROCm base is retained unchanged with its
original layers and bundled notices.

The optional archive builder uses the official Google go-containerregistry
`crane` tool during the build. The tool is not included in the submission image.
Its exact release and verified release asset SHA256 are recorded in the build
report. Original source: https://github.com/google/go-containerregistry

This notice does not grant a license to the participant's original application
code or to the competition's sample documents.
