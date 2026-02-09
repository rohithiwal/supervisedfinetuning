# Supervised Fine-Tuning (SFT) with TinyLlama

## Project Overview
This project demonstrates how to fine-tune the **TinyLlama-1.1B** model using **LoRA (Low-Rank Adaptation)** on custom datasets. It covers two main stages:
1.  **SFT on PDF Text**: Extracting and cleaning text from a PDF ("The Silent Patient") to fine-tune the base model for domain adaptation.
2.  **Instruction Tuning**: Further fine-tuning the model on a mental health counseling conversation dataset to improve its ability to follow instructions and provide empathetic responses.

## Prerequisites
Install the necessary Python libraries:

```bash
pip install pymupdf
pip install -q transformers peft bitsandbytes trl accelerate datasets
```

## Data Cleaning
The project includes a robust raw text extraction and cleaning pipeline for PDFs using `pymupdf` (fitz). The `extract_and_clean_text` function handles:
-   **Structural Cleaning**: Removes headers, footers, page numbers, and likely captions (figures/tables).
-   **Content Filtering**: Strips citations (e.g., `[12]`, `(Author et al.)`), bibliographies, and equations.
-   **Noise Reduction**: Removes academic filler phrases ("we observe that"), benchmark scaffolds ("PROMPT", "Instruction"), and low-information density blocks.
-   **Formatting**: Standardizes whitespace and removes broken sentences.

## Fine-Tuning (LoRA)
The first stage fine-tunes the model on the cleaned PDF text.

-   **Base Model**: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
-   **Quantization**: 8-bit loading via `bitsandbytes` for memory efficiency.
-   **LoRA Configuration**:
    -   **Rank (r)**: 64
    -   **Alpha**: 16
    -   **Dropout**: 0.1
    -   **Target Modules**: `q_proj`, `v_proj`
-   **Training**:
    -   Epochs: 15
    -   Batch Size: 1 (with gradient accumulation steps = 8)
    -   Learning Rate: 2e-4

## Instruction Tuning
The second stage performs instruction-based SFT using the `Amod/mental_health_counseling_conversations` dataset.

-   **Dataset**: [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
-   **Prompt Format**:
    ```
    [INST] {Context} [/INST] {Response}
    ```
-   **LoRA Configuration**:
    -   **Rank (r)**: 8
    -   **Alpha**: 16
    -   **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
-   **Training**:
    -   Epochs: 3
    -   Learning Rate: 2e-4

## Conclusion
The model's performance was evaluated at three distinct stages, showing progressive improvement:

1.  **Base Model**: The base TinyLlama model lacked specific knowledge of the text, resulting in hallucinations and vague answers irrelevant to the story.
2.  **Post-SFT (Domain Adaptation)**: After fine-tuning on the PDF text, the model successfully recognized characters and plot details (e.g., identifying Alicia Berenson's husband and his profession), though the response structure was still raw.
3.  **Post-Instruction Tuning**: Following the instruction tuning phase, the model not only retained the domain knowledge but also provided structured, coherent, and contextually accurate responses, correctly describing relationships and professions with improved fluency.