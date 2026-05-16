## 🧠 Model Architecture & Dataset

In this project, we utilized the **HyenaDNA** and **ProtMamba** foundation models, specifically employing the 1M context-length variant of HyenaDNA. The representations extracted from these State Space Models (SSMs) were fed into an attention-based decoder to predict the codon sequence. 

To optimize species-conditioned generation, our architecture features:
* **Species-Specific Heads:** A dedicated prediction head for each target species.
* **Auxiliary Heads:** Additional layers implemented for organism classification and protein abundance (PA) class prediction.

### 📊 Training Data
The model was trained on a highly curated dataset of **42,000 samples**, spanning 17 distinct species across all three domains of life:
* 🦠 **10 Bacteria**
* 🧬 **5 Eukaryota**
* 🌋 **2 Archaea**
*  For further details regarding the dataset, please refer to the Zenodo repository: https://zenodo.org/records/20131143.

<img width="754" height="412" alt="image" src="https://github.com/user-attachments/assets/81bdadbe-3572-47de-8dc5-3af76c7c9483" />


Getting Started on SLURM

To ensure the scripts execute correctly on your SLURM cluster, please configure the following settings before running the code.

### 1. Shell Script Configuration (`*.sh` files)
Open the bash scripts and update the following variables to match your specific cluster environment:
* **`ACCOUNT & PRIVILEGES`:** Replace my account details with your own.
* **`PARTITION` & `QoS`:** Specify your cluster's GPU partition and Quality of Service.
* **`MY_CACHE_DIR`:** Set the path to your working directory for caching. *(Note: Ensure you have sufficient storage space allocated here!)*
* **`MY_PROJECT_DIR`:** Set this to the absolute path of the project's root directory.
* **`WANDB_API_KEY`:** Insert your Weights & Biases API key.
* **Conda Environment:** Ensure your environment is built using the provided `environment.yml` and `requirements.txt`. Update the scripts with your exact Conda environment name.

### 2. Directory Structure & Setup (`*.py` files)
**Important:** All executable `*.py` and `*.sh` files must be placed directly in the root directory of your project. 

Your project folder must strictly adhere to the following structure:

```text
MY_PROJECT_DIR/
├── train.py                 # (Place all .py and .sh scripts here in the root)
├── data/                    
│   └── (Download the train/test/val datasets from Zenodo: [https://zenodo.org/records/20131143](https://zenodo.org/records/20131143))
├── configs/                 
│   └── (Place your configuration files here. Note: similar config files may also exist in the runs folder)
├── runs/                    
│   └── prot_mamba_pre_trained_32_r/  # (Place your training run output/weight folders here)
└── models/                  
    ├── hyena_dna/           
    │   ├── HYENA_DNA_weights.ckpt    # (Download the 1M context weights from HuggingFace)
    │   ├── modeling_hyena.py         # (Included in this repo)
    │   └── config.json               # (Included in this repo)
    └── protmamba/           
        ├── pytorch_model.bin         # (Download weights from the official ProtMamba repo)
        ├── prot_mamba_modules.py     # (Included in this repo)
        └── config.json               # (Included in this repo)

```

### ⚙️ Training from Scratch with a Different Configuration

If you want to train a model from scratch, simply follow these steps:

1. **Create a Config:** Create a new configuration file inside the `/configs` directory.
2. **Set the ID:** Inside that file, set the `experiment_id` to the name of the folder you want automatically created in the `/runs` directory.
3. **Update the Script:** Update your `.sh` execution script to point to this new config file.


## 📌 Conclusion & Future Work

In summary, this project serves as a proof of concept for the integration of DNA data and **State Space Models (SSMs)** into specific biological prediction tasks. 

It demonstrates the utility of harnessing the linear-time complexity and high expressivity of SSMs to extract meaningful features from long-sequence modalities such as DNA.

**🚀 Future Work:**
* **Dataset Expansion:** Larger datasets should be utilized to enhance the extraction of these critical features. 
* **Multimodal Injection:** Ultimately, these learned representations could later be injected into other modalities, providing valuable orthogonal information without requiring the original DNA sequence during inference.

## References & Acknowledgments

This project builds upon the foundational architectures and research from the following papers. If you find this repository useful, please consider citing the original authors:

* **HyenaDNA:** Nguyen, E., Poli, M., Faizi, M., et al. (2023). *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*. arXiv preprint. [View Paper (arXiv:2306.15794)](https://arxiv.org/abs/2306.15794)
* **ProtMamba:** Sgarbossa, D., Malbranke, C., & Bitbol, A.-F. (2024). *ProtMamba: a homology-aware but alignment-free protein state space model*. bioRxiv. [View Paper (bioRxiv)](https://www.biorxiv.org/content/10.1101/2024.05.24.595730)

  
## ✉️ Contact

For any additional information or questions regarding this project, please feel free to reach out:
* **Email:** [tallilo@mail.tau.ac.il](mailto:tallilo@mail.tau.ac.il)
