# **View Reviews** 

## **Paper ID** 

64 

## **Paper Title** 

BenchmarkHAR: A Configurable, Reproducible, and Extensible Framework for Standardized HAR Evaluation 

## **Track Name** 

Regular 

**Reviewer #1** 

## **Questions** 

## **1. Decision** 

MAJOR revision then accept after re-checking 

## **2. What category best fits this paper?** 

Research paper 

## **3. Review comments** 

Thank you for your work. I have the following comments: 

1. It is claimed benchmarking farmework but no standard benchmarking protocol is mentioned. 

2. No comparison with any other benchmark method is done. 

3. Literature review lacks benchmark studies. 

4. Not enough reference. 

## **4. Revision requirements** 

Please do the following modifications: 

1. It is claimed benchmarking farmework but no standard benchmarking protocol is mentioned. Please mention the protocol 

clearly. 

2. No comparison with any other benchmark method is done. Add at least 4 comparison. 

3. Literature review lacks benchmark studies. Add bench mark studies and their lacking. 

4. Not enough reference. Add at least 10 more reference. 

5. Increase text size in image. 

6. Add benchmarking validation. 

**5. Your level of expertise on this subject** 

Knowledgeable 

### **Reviewer #2** 

## **Questions** 

## **1. Decision** 

REJECT 

## **2. What category best fits this paper?** 

Research paper 

## **3. Review comments** 

I do not see anything novel in the approach presented in this paper. In general, the methods used are already standard practice in human activity recognition learning processes; all stages involve common techniques, and no actual innovation has been introduced. This approach should be comparable with other existing approaches as a baseline, for example BenchHAR mentioned in related work. 

- The configuration and integration of Machine Learning and Deep Learning should be clearly distinguished within the framework, as they are two distinct concepts. 

- Figures 4 and 5 lack consistency in their results. For instance, on page 9, the caption for Figure 4 indicates that the results are from 

the PAMAP2 dataset; however, a closer examination reveals that they actually correspond to the Capture-24 dataset. As an example, the performance of logistic regression on the PAMAP2 dataset should be approximately 79%, yet the value plotted in Figure 4 is around 53%. 

- In tables 1 and 2, there is a discrepancy between the number of PAMAP2 activities used. Is it 18 or 24? This raises questions about the evaluation. 

- The division of training, validation, and testing based on the number of subjects is not clearly displayed. 

- It is unclear how many features were used in the extraction stage and the selection method, particularly in the performance comparison between logistic regression and random forest. 

## **4. Revision requirements** 

- This technique is already generally used in human activity recognition. 

- Please check the review comments in detail. 

**5. Your level of expertise on this subject** 

Knowledgeable 

### **Reviewer #3** 

## **Questions** 

## **1. Decision** 

Soft reject but POSITION paper presentation after editing 

## **2. What category best fits this paper?** 

Research paper 

## **3. Review comments** 

BenchmarkHAR addresses a relevant problem: inconsistent 

evaluation practices in wearable HAR. The configurable pipeline and integration of classical, deep-learning, and pretrained models could provide a useful community tool. However, the current experiments primarily demonstrate successful execution, rather than adequately validating the framework’s claims regarding fair comparison, leakage prevention, and reproducibility. 

The experiments in Section V-A are explicitly described as unsuitable for quantitative model comparison, yet subsequent sections draw conclusions about relative performance and computational efficiency. The short training schedules do not sufficiently support these conclusions. Furthermore, Figures 4–5 are labelled as PAMAP2 results, but their macro-F1 values appear to correspond to Capture-24 in Table III. The accompanying interpretation also conflicts with the plots. These discrepancies undermine the performance–complexity analysis and require verification against the underlying results. Overall, I see value in a position-paper presentation, but insufficient evidence for fullpaper acceptance in the current form. 

## **4. Revision requirements** 

1. Clarify the scope. Present the work as an initial framework with preliminary execution validation. Remove or qualify unsupported model-ranking, efficiency, and convergence claims, and distinguish implemented capabilities from experimentally validated guarantees. 2. Correct the results and their interpretation. Verify Figures 4–5 against the underlying result files and reconcile the figures, tables, and text, including the WISDM model ranking and the definition of BioPM training time. Remove analyses that cannot be reliably corrected. 

3. Complete the experimental specification. Document the classical features, input channels, actual activity classes, subject-split proportions, and Capture-24 subset-selection procedure. Clarify 

how preprocessing and model selection respect training/test boundaries, and provide clear instructions and configuration references for reproducing the reported runs. 

## **5. Your level of expertise on this subject** 

Expert 

### **Reviewer #4** 

## **Questions** 

## **1. Decision** 

Accept with minor revision (please provide a short list of edits) 

## **2. What category best fits this paper?** 

Research paper 

## **3. Review comments** 

The paper “Benchmark HAR: A Configurable, Reproducible, and Extensible Framework for Standardized HAR Evaluation” presents a framework designed to standardize the evaluation of wearablebased human activity recognition (HAR) methods. The framework supports a wide range of approaches, including traditional machine learning algorithms, deep learning architectures, and more recent pretrained foundation models. 

Overall, I consider this an excellent paper. The work addresses an important challenge in HAR research, where differences in experimental protocols often make comparisons between methods difficult or even misleading. I particularly appreciate the authors' emphasis on reproducibility and methodological rigor, especially their strict application of leave-one-subject-out (LOSO) evaluation to prevent data leakage and ensure a more realistic assessment of generalization. The methodology is clearly described, and the paper 

is well structured and well written. 

I have only a few relatively minor concerns. 

First, the selection of deep learning architectures is not entirely justified. In particular, I find it surprising that no Transformer-based architecture was included, given its relevance in contemporary HAR research. Additionally, the explanation distinguishing the RNN and LSTM implementations is somewhat confusing. The authors should clarify this section, as the current description does not clearly establish the intended architectural differences. 

Second, I have some concerns regarding the computational scalability of the proposed framework. The authors acknowledge having to reduce the size of the Capture-24 dataset due to computational constraints. While understandable, this raises questions about the practicality of extending the framework to more computationally demanding foundation models. Since these models can be significantly more expensive to train, fine-tune, or evaluate, it would be useful to discuss the expected computational requirements and limitations of the framework when handling larger datasets and more complex models. 

Despite these concerns, I believe the paper makes a valuable contribution to the HAR community. The framework addresses a genuine need for standardized and reproducible evaluations, and the authors demonstrate a strong understanding of the methodological challenges involved. The limitations mentioned above are relatively minor and do not diminish the overall quality of the work. 

## **4. Revision requirements** 

Clarify the RNN vs LSTM part. If you can, enhance the discussion 

regarding the impact of using the framework with much larger foundation models. 

## **5. Your level of expertise on this subject** 

Expert 

