# DuDel: Exon-Level CNV Detection from Whole Exome Sequencing Data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20431569.svg)](https://doi.org/10.5281/zenodo.20431569)

**DuDel v1.0.0** is a machine learning–based exon-specific CNV (Copy Number Variation) caller that uses Balanced Random Forest classification to identify deletions and duplications from WES-derived exon-level read counts.
Developed by the *Clinical Omics & Informatics Unit (COIN), Neuroscience Institute, University of Cape Town, South Africa.*

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Structure](#pipeline-structure)
3. [Installation](#installation)
4. [Input Requirements](#input-requirements)
5. [Usage](#usage)
6. [Output Files](#output-files)
7. [Scripts Overview](#scripts-overview)

---

## Overview

**DuDel** integrates multiple scripts to perform the following steps:

1. **Generate exon-level BED file** from a reference GFF and gene list.
2. **Compute exon-level read counts** from BAM files using `bedtools coverage`.
3. **Normalize read counts and compute reference correlations** in R.
4. **Predict exon-level CNVs** using a pre-trained Random Forest model (`.pkl` file).
5. Optionally **annotate** results with phenotype, ClinGen, and predictive gene/protein scores, and **export** results as `.csv`, `.vcf`, and text summary reports.

---

## Pipeline Structure

```
DuDel/
│
├── RFmodel/                     # Pre-trained Random Forest model
├── datasets/                    # Annotation datasets (ClinGen, gene/protein scores, etc.)
├── panels/                      # Example disease-specific panels
├── 1-generate-exon-level-bed-file.sh
├── 2-exon-level-counts.sh
├── 3-Count-Matrix.R
├── dudel.py
├── dudel_RF.py
├── README.md
└── dudel_env.yml
```

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f dudel_env.yml
conda activate dudel_env
```

Ensure `bedtools` and `R` are installed and accessible from your `$PATH`.

Download the GFF annotation file

```bash
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gff.gz
```

Download the Model file

```bash
wget https://zenodo.org/records/20431569/files/dudel_brf.joblib -P RFmodel/
```

---

## Input Requirements

| Input Type      | Description                                               | Example                      |
| --------------- | --------------------------------------------------------- | ---------------------------- |
| **GFF File**    | Reference genome annotation file containing exon features | `GRCh38.gff3`                |
| **Gene List**   | Plain text file with one gene symbol per line             | `genes_list.txt`             |
| **BAM Files**   | Aligned sequencing reads for reference and test samples   | `sample1.bam`, `sample2.bam` |
| **Model File**  | Pre-trained Random Forest model (`.joblib`)                  | `RFmodel/DuDel_brf.joblib` |
| **Annotations** | Optional datasets (ClinGen, gene/protein scores)          | `datasets/`                  |

---

## Usage

### Step 1: Generate Exon-Level BED File

```bash
bash 1-generate-exon-level-bed-file.sh <GFF_FILE> <GENES_LIST> <OUTPUT_PREFIX>
```

**Example:**

```bash
bash 1-generate-exon-level-bed-file.sh GCF_000001405.40_GRCh38.p14_genomic.gff hg38CodingGenes.txt GenePanel
```

Output: `GenePanel.bed`

---

### Step 2: Generate Exon-Level Read Counts

```bash
bash 2-exon-level-counts.sh <REF_BAMS_LIST> <TEST_BAMS_LIST> <BED_FILE>
```

**Example:**

```bash
bash 2-exon-level-counts.sh refBams.txt testBam.txt GenePanel.bed
```

Outputs:

```
./ReferencesCounts/SampleID.txt
./TestCounts/SampleID.txt
```

---

### Step 3: Compute Count Matrices and Normalization

```bash
Rscript 3-Count-Matrix.R ./TestCounts ./ReferencesCounts ./OutputDir
```

Output: `<sample_id>_count_matrix.csv` for each test sample.

---

### Step 4: Run DuDel CNV Prediction

```bash
python dudel.py \
  -e <EXON_COUNT_MATRIX> \
  -b <EXON_BED_FILE> \
  -m <RANDOM_FOREST_MODEL> \
  -o <OUTPUT_DIRECTORY> \
  [optional flags]
```

**Optional flags:**

| Flag | Description                   |
| ---- | ----------------------------- |
| `-p` | Add phenotype annotations     |
| `-v` | Generate VCF file             |
| `-c` | Add ClinGen annotations       |
| `-G` | Add gene predictive scores    |
| `-P` | Add protein predictive scores |
| `-s` | Generate summary report       |

**Example:**

```bash
python3 dudel.py \
  -e ./OutputDir/sample1_count_matrix.csv \
  -b GenePanel.bed \
  -m ./RFmodel/dudel-smote-icgnmd.pkl \
  -o ./CNV_Results \
  -p -c -G -P -s -v
```

<img width="2555" height="701" alt="image" src="https://github.com/user-attachments/assets/11abe364-94cd-48bc-aee7-f5ab01eea2ad" />


---

## Output Files

| Output                | Description                       |
| --------------------- | --------------------------------- |
| `*.dudel.csv`         | Full CNV prediction table         |
| `*.dudel.cnv.vcf`     | VCF file (if `-v` used)           |
| `*.SummaryReport.txt` | Detailed summary (if `-s` used)   |
| `*.count_matrix.csv`  | Normalized exon-level read counts |

---

## Scripts Overview

### **1-generate-exon-level-bed-file.sh**

Generates exon-level BED file for selected genes from a GFF file.
**Inputs:** GFF file, gene list.
**Outputs:** BED file containing exon coordinates per gene.

### **2-exon-level-counts.sh**

Computes exon-level coverage counts using `bedtools coverage` for reference and test BAM files.
**Outputs:** Coverage files in `./ReferencesCounts` and `./TestCounts`.

### **3-Count-Matrix.R**

* Reads exon count data.
* Performs normalization (RPKM) using `edgeR`.
* Computes exon-level correlations between test and reference samples.
* Generates combined count matrices and writes normalized outputs.

### **dudel.py**

The main CNV prediction tool.

* Loads normalized exon-level data.
* Applies a trained Random Forest model.
* Identifies deletions and duplications.
* Optionally adds phenotype and functional annotations.
* Generates tabular, VCF, and summary outputs.

---
DuDel v1.0 – Exon-Specific CNV Caller from WES Data.
Clinical Omics & Informatics Unit (COIN), Neuroscience Institute, University of Cape Town, South Africa.

