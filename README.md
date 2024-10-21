
# BiLSTM-CRF Named Entity Recognition (NER) using BERT

Code for the conference paper "A HYBRID BILSTM-CRF MODEL WITH BERT EMBEDDINGS FOR NAMED ENTITY RECOGNITION ON CONFIDENTIAL SYNTHETIC DATA", which is published in V. International Science and Innovation Congress.

This project implements a hybrid Named Entity Recognition (NER) model using a combination of a BiLSTM-CRF architecture with BERT embeddings. The model is designed to identify and label entities such as names (PERSON), addresses (ADDRESS), and email addresses (EMAIL) in text data. The project also includes evaluation and comparison with spaCy’s NER model, both pre-trained and custom versions.

## Project Components

- **BiLSTM-CRF Model**: A neural network that combines BERT embeddings with a BiLSTM-CRF architecture to perform sequence labeling.
- **Data Generation**: We use the Faker library to generate a large synthetic dataset containing person names, addresses, and email addresses.
- **Training & Evaluation**: The model is trained and evaluated using this synthetic data, with comparisons to spaCy’s NER models.
- **Model Saving**: The trained hybrid model can be saved and loaded for inference on new data.
- **Evaluation & Comparison**: The project evaluates and compares the hybrid model's performance against spaCy's pre-trained and custom models, if available.

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed, and install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `torch`
- `transformers`
- `spacy`
- `scikit-learn`
- `Faker`
- `TorchCRF`

You will also need to download the `en_core_web_sm` model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

### Dataset Generation

We use the `Faker` library to create synthetic data for training and testing. Each data point contains sentences mentioning person names, addresses, and email addresses with their corresponding character offsets for entity recognition.

### Model Architecture

The NER model consists of the following components:

- **BERT Embeddings**: The `bert-base-uncased` model is used to extract contextualized embeddings for each token in the input sequence.
- **BiLSTM Layer**: A BiLSTM layer processes the BERT embeddings to capture sequential dependencies in the text.
- **CRF Layer**: A CRF layer is used to predict the entity labels and enforce valid label transitions.
  
### Training

The model is trained using synthetic data containing 20,000 sentences. During training, the model learns to recognize and label `PERSON`, `ADDRESS`, and `EMAIL` entities using BIO tagging.

This script includes:

- Data generation using Faker.
- Model training over 55 epochs with a learning rate of `2e-5`.
- Batch size of 32.

### Evaluation

The project includes evaluation scripts that test the model on 100 generated paragraphs and compare it with:

- **Original spaCy Model** (`en_core_web_sm`)
- **Custom spaCy Model** (if available)

This script will print classification reports for each model, comparing precision, recall, and F1-scores across entities (PERSON, ADDRESS, EMAIL).

## Model Inference

Once the model is trained, you can load it and perform inference on new text:

```python
import torch

model = BiLSTM_CRF('bert-base-uncased', 128, len(LABEL_MAP))
model.load_state_dict(torch.load('hybrid_ner_model.pt'))
```

You can then input text and get entity predictions from the hybrid model.

### Example Inference

Sample output from the hybrid model and spaCy models is printed for comparison, highlighting their differences in detected entities.

## Results

The classification reports include:

- **Precision**: The ratio of true positive predictions over all positive predictions.
- **Recall**: The ratio of true positive predictions over all actual positives.
- **F1-score**: The harmonic mean of precision and recall.
- **Support**: The number of occurrences of each label in the evaluation set.

The hybrid model performs comparably to spaCy’s models, with potential for fine-tuning on real-world data.

## Custom spaCy Model

If a custom spaCy model is available (e.g., trained on domain-specific data), the script will load and evaluate it alongside the original spaCy model and the hybrid BiLSTM-CRF model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
