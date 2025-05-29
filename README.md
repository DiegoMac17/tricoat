# Tricoat

This repository contains the Tricoat project.  
Public version in construction.

Publication can be found at: https://www.mdpi.com/2075-4426/14/4/421

## Description

This repository implements **Tri-COAT** (Tri-Modal Co-Attention Transformer), a multimodal deep learning framework for early-stage, progression-specific Alzheimer's disease subtyping. Tri-COAT explicitly fuses imaging (MRI-derived quantitative traits), genetic (SNP dosage, odds ratios, allele frequencies, chromosome embeddings), and clinical assessment data via a tri-modal co-attention mechanism to learn cross-modal feature interactions and provide interpretable predictions. On the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, Tri-COAT achieves state-of-the-art classification AUROC while highlighting key biomarker associations :contentReference[oaicite:0]{index=0}.

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/tricoat.git
   cd tricoat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the data**
    Data can be downloaded from the [Alzheimer's Disease Neuroimaging Initiative (ADNI) repository](https://adni.loni.usc.edu/data-samples/accessing-adni-data/).

    The data should be organized as follows:
    ```
    data/
    imaging/    # CSV or NumPy arrays of ROI traits
    genetics/   # CSV with SNP dosage + metadata
    clinical/   # CSV with clinical scores
    ```
## Usage

4. **Run the model**
    ```bash
    python train.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact us at [machad@rpi.edu](mailto:machad@rpi.edu).
