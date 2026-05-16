In this project, we utilized the HyenaDNA and ProtMamba foundation models, specifically employing the 1M context length version of HyenaDNA. The outputs from these State Space Models (SSMs) were fed into an attention-based decoder to predict the codon sequence. To enhance species-conditioned generation, the decoder featured a dedicated prediction head per species.
Auxiliary heads for both organism classification and protein abundance class prediction were also implemented.

Only 42,000 samples were used for training, spanning 17 distinct species from all three domains of life: 10 Bacteria, 5 Eukaryota, and 2 Archaea. For further details regarding the dataset, please refer to the Zenodo repository: https://zenodo.org/records/20131143.

<img width="754" height="412" alt="image" src="https://github.com/user-attachments/assets/81bdadbe-3572-47de-8dc5-3af76c7c9483" />



To ensure the scripts execute correctly on your SLURM server, please note the following in the files:

In the *.sh files you should:
  -Put your account in ACCOUNT & PRIVILEGES insted of mine.
  -The QoS you use and the PARTITION you use ( GPU ) .
  -Make sure to put you working dir path for all the cash in "MY_CACHE_DIR" (make sure you have enogh space there!!!)
  - Make sure to put your home directory of the project in "MY_PROJECT_DIR"
  - In WANDB_API_KEY put your wandb key
  - make sure you condat env set up with environment.yml , requirements.txt . Change to you conda env name.
  - make sure you have configs file in you home directory of the project when the configs files inside it.

In the *.py files you should:
  - make sure you have runs, configs , models , data folders
  - In the /runs folder you should put the weight including folder ( for instance all the folder "prot_mamba_pre_trained_32_r" )
  - in the /configs you should put the configs files , be carful, similar config file exist also in the weight including folder.
  - in the /models you should have the dirs: hyena_dna, protmamba
  - inside /models/hyena_dna you should have : HYENA_DNA_weights.ckpt - the wights for the 1M context Hyena DNA model from hugginface, and the files modeling_hyena.py , config.json that are in this git repo.
  - inside /models/protmamba  you should have : pytorch_model.bin - the weights of prot Mamba from thier git repo , and the files: prot_mamba_modules.py , config.json  that are  in this git repo.
  - inside the /data folder , you shoud have the datasets for train ,test and validation that uploded to Zenodo : https://zenodo.org/records/20131143

Make sure to put the *.py and *.sh files in the home dir of you project!



In summary, this project serves as a proof of concept for the integration of DNA data and State Space Models (SSMs) into specific biological prediction tasks. It demonstrates the utility of harnessing the linear-time complexity and high expressivity of SSMs to extract meaningful features from long-sequence modalities such as DNA. Future work should utilize larger datasets to enhance the extraction of these critical features. Ultimately, these learned representations could later be injected into other modalities, providing valuable orthogonal information without requiring the original DNA sequence during inference.

## References & Acknowledgments

This project builds upon the foundational architectures and research from the following papers. If you find this repository useful, please consider citing the original authors:

* **HyenaDNA:** Nguyen, E., Poli, M., Faizi, M., et al. (2023). *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*. arXiv preprint. [View Paper (arXiv:2306.15794)](https://arxiv.org/abs/2306.15794)
* **ProtMamba:** Sgarbossa, D., Malbranke, C., & Bitbol, A.-F. (2024). *ProtMamba: a homology-aware but alignment-free protein state space model*. bioRxiv. [View Paper (bioRxiv)](https://www.biorxiv.org/content/10.1101/2024.05.24.595730)

* 
For any additional information:
tallilo@mail.tau.ac.il
