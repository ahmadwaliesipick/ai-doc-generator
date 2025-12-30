# Work Plan for the Research

## Phase 1: Literature Review & Problem Refinement (Weeks 1--2)

-   Review key research papers on CodeT5, CodeBERT, code summarization,
    and evaluation metrics.
-   Finalize research title, scope, and objectives.

**Deliverable:** Updated literature review and finalized objectives.

## Phase 2: Dataset Collection & Preparation (Weeks 3--5)

-   Select 200--500 GitHub repositories (JavaScript & PHP).
-   Extract function--comment pairs.
-   Clean data (remove duplicates, trivial comments).
-   Split dataset into training, validation, and test sets.

**Deliverable:** Clean dataset in CSV format.

## Phase 3: Baseline Implementation (Weeks 6--7)

-   Implement baseline models:
    -   Pre-trained CodeT5 (no fine-tuning)
    -   Optional keyword-based baseline (RAKE)

**Deliverable:** Baseline evaluation results.

## Phase 4: Model Fine-Tuning (Weeks 8--10)

-   Fine-tune CodeT5 using HuggingFace.
-   Monitor training and validation performance.

**Deliverable:** Fine-tuned CodeT5 model.

## Phase 5: Automatic Evaluation (Weeks 11--12)

-   Evaluate generated comments using BLEU, ROUGE, and METEOR.

**Deliverable:** Quantitative evaluation tables.

## Phase 6: Human Evaluation (Weeks 13--14)

-   Conduct human evaluation with developers.
-   Assess correctness, readability, and usefulness.

**Deliverable:** Human evaluation results.

## Phase 7: Analysis & Discussion (Weeks 15--16)

-   Analyze results and compare baselines.
-   Discuss findings and limitations.

**Deliverable:** Results and discussion chapter.

## Phase 8: Final Documentation & Submission (Weeks 17--18)

-   Complete thesis writing.
-   Review with supervisor.
-   Submit final thesis.

**Deliverable:** Final thesis submission.
