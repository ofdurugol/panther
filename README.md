# [MICCAI2025 Panther] Multi-Stage Fine-Tuning and Ensembling for Pancreatic Tumor Segmentation in MRI

This repository contains the official implementation of our MICCAI 2025 challenge paper:

### **A Multi-Stage Fine-Tuning and Ensembling Strategy for Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI**

We present a **cascaded pre-training, fine-tuning, and ensembling framework** for **pancreatic ductal adenocarcinoma (PDAC) segmentation** in MRI. Our method, built on **nnU-Net v2**, leverages:  
- Multi-stage cascaded fine-tuning (from a foundation model → CT lesions → target MRI modalities)  
- Systematic augmentation ablations (aggressive vs. default)  
- Metric-aware heterogeneous ensembling ("mix of experts")  

This approach achieved **the first place** in the **PANTHER challenge**, with robust performance across both **diagnostic T1W (Task 1)** and **therapeutic T2W MR-Linac (Task 2)** scans.

> **Authors**: Omer Faruk Durugol*, Maximilian Rokuss*, Yannick Kirchhoff, Klaus H. Maier-Hein  
> *Equal contribution  
> **Paper**: [![arXiv](https://img.shields.io/badge/arXiv-2508.21775-b31b1b.svg)](https://arxiv.org/abs/2508.21775)  
> **Challenge**: [PANTHER](https://panther.grand-challenge.org)

---

## News/Updates
- 🏆 **Aug 2025**: Achieved **1st place in both tasks** in the [PANTHER Challenge](https://panther.grand-challenge.org)!  
- 📄 **Aug 2025**: Paper preprint released on [arXiv](https://arxiv.org/abs/2508.21775).  

---

## 🚀 Usage

Coming soon!


## 📂 Data

We trained and evaluated on the **PANTHER Challenge dataset**:
👉 [https://zenodo.org/records/15192302](https://zenodo.org/records/15192302)

Pretraining leveraged:

* [MultiTalentV2](https://zenodo.org/records/13753413)
* Pancreatic lesion CT datasets (MSD + PANORAMA)

---

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{durugol2025multistagefinetuningensemblingstrategy,
      title={A Multi-Stage Fine-Tuning and Ensembling Strategy for Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI}, 
      author={Omer Faruk Durugol and Maximilian Rokuss and Yannick Kirchhoff and Klaus H. Maier-Hein},
      year={2025},
      eprint={2508.21775},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21775}, 
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please reach out:
📧 [maximilian.rokuss@dkfz-heidelberg.de](mailto:maximilian.rokuss@dkfz-heidelberg.de)