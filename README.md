<div align="center">
<img src="assets/codefuse_logo.png" width="70%">
</div>


# SVRepair: Structured Visual Reasoning for Code Repair

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Static Badge](https://img.shields.io/badge/Paper-arxiv.2602.06090-B31B1B.svg)](https://arxiv.org/abs/2602.06090)



Large language models (LLMs) have recently shown strong potential for Automated Program Repair (APR), yet most existing approaches remain unimodal and fail to leverage the rich diagnostic signals contained in visual artifacts such as screenshots and control-flow graphs. In practice, many bug reports convey critical information visually (e.g., layout breakage or missing widgets), but directly using such dense visual inputs often causes context loss and noise, making it difficult for MLLMs to ground visual observations into precise fault localization and executable patches. To bridge this semantic gap, we propose **SVRepair**, a multimodal APR framework with structured visual representation. SVRepair first fine-tunes a vision-language model, **Structured Visual Representation (SVR)**, to uniformly transform heterogeneous visual artifacts into a *semantic scene graph* that captures GUI elements and their structural relations (e.g., hierarchy), providing normalized, code-relevant context for downstream repair. Building on the graph, SVRepair drives a coding agent to localize faults and synthesize patches, and further introduces an iterative visual-artifact segmentation strategy that progressively narrows the input to bug-centered regions to suppress irrelevant context and reduce hallucinations. Extensive experiments across multiple benchmarks demonstrate state-of-the-art performance: SVRepair achieves **36.47%** accuracy on SWE-Bench M, **38.02%** on MMCode, and **95.12%** on CodeVision, validating the effectiveness of SVRepair for multimodal program repair.

📦 **Model Weights**: [CodeFuse-SVR-8B](http://huggingface.co/codefuse-ai/CodeFuse-SVR-8B)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Agent: [CodeFuse-Agent](https://github.com/codefuse-ai/CodeFuse-Agent.git)
- SWE-Bench-Multimodel: [SWE-bench Multimodal](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal)


### Installation


```bash
# Clone repository
git clone https://github.com/codefuse-ai/CodeFuse-SVR.git
cd CodeFuse-SVR

# Install dependencies
export PYTHONPATH=./
pip install -r requirements.txt
```

### Dataset Setup
Process SWE-bench Multimodal data and download repositories

```bash
# output.json will be used as input_data in subsequent steps
python src/utils/process_data_dl_repo.py --parquet-path /path/to/your/test-00000-of-00001.parquet --output-path /path/to/your/output.json --repo-dir /path/to/your/repo
```

### Run Complete Pipeline

Use the provided script:

```bash
# Run full SVRepair pipeline
bash scripts/full_run.sh
```

## ⚙️ Configuration

### Environment Variables
```bash
# API Configuration
export OPENAI_API_KEY="your-agent-api-key"
export VLM_API_KEY="your-vlm-api-key"

# System Configuration
export PYTHONPATH=./
```

### Input Data Format
```json
{
  "grommet__grommet-6282": {
        "repo": "grommet/grommet",
        "instance_id": "grommet__grommet-6282",
        "base_commit": "xxxxx",
        "patch": "",
        "test_patch": "",
        "problem_statement": "Bug description...",
        "hints_text": "",
        "created_at": "",
        "image_assets": "{\"problem_statement\": [\"url/of/image1.png\", \"url/of/image2.png\"]}",
        "version": "",
        "FAIL_TO_PASS": "",
        "PASS_TO_PASS": ""
    }
}
```


## 📋 Command Line Interface

SVRepair provides a comprehensive CLI with the following commands:

### 1. Full Pipeline (`full-run`)
Run the complete SVRepair workflow:
```bash
python main.py full-run \
  --input_data PATH \
  --output_dir PATH \
  --repo_path PATH \
  --image_dir PATH \
  --vlm_model MODEL \
  --vlm_url URL \
  --model_name MODEL \
  --base_url URL \
  [--temperature 0.0] \
  [--max_workers 4] \
  [--copy_repo] \
  [--project_name svrepair]
```

### 2. Generate Image IR (`generate-image-ir`)
Generate structured representations from visual artifacts:
```bash
python main.py generate-image-ir \
  --model_name MODEL \
  --base_url URL \
  --input_data PATH \
  --image_dir PATH \
  --result_path PATH \
  --output_dir PATH \
  [--max_workers 4]
```

### 3. Generate Patches (`generate-patch`)
Generate code patches based on image IR:
```bash
python main.py generate-patch \
  --image_ir_path PATH \
  --output_dir PATH \
  --repo_path PATH \
  --model_name MODEL \
  --base_url URL \
  [--temperature 0.0] \
  [--copy_repo]
```

### 4. Validate Patches (`validation`)
Validate generated patches:
```bash
python main.py validation \
  --image_ir_path PATH \
  --result_path PATH \
  --output_dir PATH \
  --model_name MODEL \
  --base_url URL \
  [--max_workers 4] \
  [--repo_path PATH]
```

### 5. Image Localization (`localization`)
Localize code segments based on visual context:
```bash
python main.py localization \
  --repo_path PATH \
  --image_dir PATH \
  --output_dir PATH \
  --model_name MODEL \
  --base_url URL \
  --result_path PATH
```

## 📁 Output Structure

After running SVRepair, you'll find the following structure:

```
results/
├── image_ir_data.json          # Image IR results
├── instance_1/            # Generated patch files
│   ├── cropped_image.png
│   ├── instance_1.patch
│   ├── resp.json
│   ├── subgraph_instance_1.json
│   └── ...
├── instance_2/
├── ...
├── swebench_image_cropped_instance.json  # All generated patches
├── all_validation_failed_instance.json # Validation results
├── all_subgraphs_merged.json # Subgraph localization results
└── SVR-VL_result_path.json  # patch-diff
```

### Output Formats

**Patch File Format:**
```diff
--- a/file/path
+++ b/file/path
@@ -10,7 +10,7 @@
- old code
+ new code
```


## 📊 Pipeline Overview

The SVRepair pipeline consists of 6 main stages:

1. **Image IR Generation**: Convert visual artifacts to structured representations
2. **Patch Generation**: Generate code patches based on IR and problem statements
3. **Validation**: Validate patches using rule-based and agent-based methods
4. **Localization**: Identify relevant code segments from visual context
5. **Refined Patch Generation**: Generate improved patches using localization results
6. **Result Processing**: Compile final results and generate evaluation files


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

```
@misc{tang2026svrepairstructuredvisualreasoning,
      title={SVRepair: Structured Visual Reasoning for Automated Program Repair}, 
      author={Xiaoxuan Tang and Jincheng Wang and Liwei Luo and Jingxuan Xu and Sheng Zhou and Dajun Chen and Wei Jiang and Yong Li},
      year={2026},
      eprint={2602.06090},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.06090}, 
}
```