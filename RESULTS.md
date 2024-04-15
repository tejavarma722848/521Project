# Results
This section presents the results obtained from the evaluation of various deep learning models for sentiment analysis on the Twitter dataset. The performance of models such as CNN, LSTM, CNN-LSTM, LSTM-CNN, ELMo, and BERT is compared using metrics including Accuracy, Precision, Recall, and F1 score.

## Model Evaluation Metrics 
The evaluation metrics used in this study are as follows:

Accuracy: Represents the ratio of correctly classified observations to the total number of observations.
Precision: Measures the proportion of positively labeled instances that were correctly predicted.
Recall: Indicates the ratio of true positive instances to the total actual positive instances.
F1 Score: A harmonic mean of precision and recall, providing a balance between the two metrics.

## Impact of Dataset Size and Model Complexity

The study examines the performance of deep learning models across varying dataset sizes and complexities. Different combinations of hyperparameters such as epochs and batch sizes are explored to understand their effects on model performance.

Due to the large size of the dataset used in this project, we have decided not to include it in the GitHub repository. Including the dataset would significantly increase the repository size, making it cumbersome to download and clone. Moreover, hosting large datasets on GitHub can lead to performance issues and may violate GitHub's guidelines on repository size limits. The dataset size exceeds the storage capacity typically provided by online hosting platforms. Uploading such a large dataset to a public repository would not be feasible.Even if we were to provide a link to an external hosting service, accessing and downloading the dataset may pose challenges for users with limited bandwidth or storage capacity.


## Summary

| Model     | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| CNN       | 0.820    | 0.804     | 0.850  | 0.826    |
| LSTM      | 0.872    | 0.863     | 0.883  | 0.873    |
| CNN-LSTM  | 0.812    | 0.811     | 0.815  | 0.813    |
| LSTM-CNN  | 0.807    | 0.824     | 0.782  | 0.803    |
| ELMo      | 0.767    | 0.727     | 0.854  | 0.786    |
| BERT      | 0.919    | 0.916     | 0.924  | 0.920    |

The performance measures of the models across dataset size of 20k:


| Model     | Accuracy | Precision | Recall | F1 Measure |
|-----------|----------|-----------|--------|------------|
| CNN       | 0.837776 | 0.82697   | 0.85326 | 0.839909   |
| LSTM      | 0.828585 | 0.803408  | 0.868924| 0.834883   |
| CNN-LSTM  | 0.829504 | 0.801817  | 0.874223| 0.836456   |
| LSTM-CNN  | 0.839154 | 0.851374  | 0.851374| 0.835796   |
| ELMo      | 0.801684 | 0.804241  | 0.792605| 0.798381   |
| BERT      | 0.948147 | 0.9456478 | 0.951034| 0.9483334  |

The performance measures of the models across dataset size of 100k:


| Model     | Accuracy | Precision | Recall | F1 Measure |
|-----------|----------|-----------|--------|------------|
| CNN       | 0.839629 | 0.870428  | 0.798466 | 0.832895   |
| LSTM      | 0.85167  | 0.866471  | 0.844841 | 0.855519   |
| CNN-LSTM  | 0.847022 | 0.830932  | 0.871751 | 0.850852   |
| LSTM-CNN  | 0.846449 | 0.850834  | 0.840605 | 0.845689   |
| ELMo      | 0.826781 | 0.850873  | 0.842645 | 0.808321   |
| BERT      | 0.97769326 | 0.9775955 | 0.97779113 | 0.97769326 |

The performance measures of the models across dataset size of 200k:


Discussion
The results indicate that BERT consistently outperforms other models across all metrics and dataset sizes. However, it is noteworthy that the performance of traditional models such as CNN and LSTM is competitive, especially considering the simplicity of their architectures compared to BERT.

