# Import necessary libraries
import random
from faker import Faker
import spacy
from sklearn.metrics import classification_report
import torch
from torch import nn
from TorchCRF import CRF
from transformers import BertTokenizerFast, BertModel
import re

# Initialize Faker and spaCy
fake = Faker()
nlp = spacy.load('en_core_web_sm')  # Load pre-trained English model

# Define BiLSTM-CRF model using BERT embeddings
class BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_labels):
        super(BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Allow fine-tuning of BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)
        if labels is not None:
            loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool(),
                reduction='mean'
            )
            return loss
        else:
            return self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )

# Parameters
BERT_MODEL_NAME = 'bert-base-uncased'
HIDDEN_DIM = 128

# Updated LABEL_MAP for BIO tagging
LABEL_MAP = {
    "B-PERSON": 0,
    "I-PERSON": 1,
    "B-ADDRESS": 2,
    "I-ADDRESS": 3,
    "B-EMAIL": 4,
    "I-EMAIL": 5,
    "O": 6,
    "PAD": 7
}
NUM_LABELS = len(LABEL_MAP)

# Reverse map for decoding labels
LABEL_MAP_REV = {v: k for k, v in LABEL_MAP.items()}

# Initialize model and tokenizer
model = BiLSTM_CRF(BERT_MODEL_NAME, HIDDEN_DIM, NUM_LABELS)
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

# Function to convert examples into BERT format with correct label assignment
def encode_examples(examples):
    input_ids_list = []
    attention_masks_list = []
    label_ids_list = []

    for sentence, annotations in examples:
        # Tokenize the sentence
        tokenized_inputs = tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        offset_mapping = tokenized_inputs.pop('offset_mapping')[0]
        special_tokens_mask = tokenized_inputs.pop('special_tokens_mask')[0]
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        
        # Initialize labels
        labels = [LABEL_MAP['O']] * len(word_ids)
        
        # Map entities to word indices
        entities = []
        for start_char, end_char, label in annotations:
            entities.append({'start': start_char, 'end': end_char, 'label': label})
        
        # Assign labels
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens
                labels[idx] = LABEL_MAP['PAD']
            else:
                token_start, token_end = offset_mapping[idx]
                token_label = 'O'
                for entity in entities:
                    if token_end <= entity['start']:
                        continue
                    if token_start >= entity['end']:
                        continue
                    if token_start == entity['start']:
                        token_label = 'B-' + entity['label']
                    else:
                        token_label = 'I-' + entity['label']
                    break  # Once the token is labeled, break out of the loop
                labels[idx] = LABEL_MAP.get(token_label, LABEL_MAP['O'])
        
        input_ids_list.append(tokenized_inputs['input_ids'])
        attention_masks_list.append(tokenized_inputs['attention_mask'])
        label_ids_list.append(torch.tensor(labels))
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    label_ids = torch.stack(label_ids_list)

    return input_ids, attention_masks, label_ids

# Prepare training data with character offsets
training_data = []
for _ in range(20000):  # Increased training size
    name = fake.name()
    address = fake.address().replace("\n", ", ")
    email = fake.email()
    sentence = f"My friend {name} lives at {address}. You can contact them at {email}."
    entities = []
    for entity_text, label in [(name, "PERSON"), (address, "ADDRESS"), (email, "EMAIL")]:
        start = sentence.find(entity_text)
        end = start + len(entity_text)
        entities.append((start, end, label))
    training_data.append((sentence, entities))

# Convert training data to BERT format
input_ids, attention_masks, label_ids = encode_examples(training_data)

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)
label_ids = label_ids.to(device)

# Train the BiLSTM-CRF model
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Adjusted learning rate
model.train()

for epoch in range(55):  # Adjust epochs as needed
    total_loss = 0
    for i in range(0, len(input_ids), 32):  # Batch size of 32
        batch_input_ids = input_ids[i:i+32]
        batch_attention_masks = attention_masks[i:i+32]
        batch_label_ids = label_ids[i:i+32]
        loss = model(batch_input_ids, batch_attention_masks, batch_label_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / (len(input_ids) / 32)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# Save the model
torch.save(model.state_dict(), "hybrid_ner_model.pt")

# Prepare test data with character offsets
test_data = []
for _ in range(100):
    sentences = []
    entities = []
    for _ in range(5):
        if random.choice([True, False]):
            name = fake.name()
            address = fake.address().replace("\n", ", ")
            email = fake.email()
            sentence = f"My colleague {name} just moved to {address}. Reach out to them at {email}."
            # Add entities to the list with character offsets
            current_len = sum(len(s) + 1 for s in sentences)  # +1 for space
            for entity_text, label in [(name, "PERSON"), (address, "ADDRESS"), (email, "EMAIL")]:
                start = current_len + sentence.find(entity_text)
                end = start + len(entity_text)
                entities.append((start, end, label))
            sentences.append(sentence)
        else:
            sentence = fake.text(max_nb_chars=100)
            sentences.append(sentence)
    paragraph = " ".join(sentences)
    test_data.append((paragraph, entities))

# Function to evaluate the hybrid model
def evaluate_hybrid_model(model, tokenizer, test_data, model_name="Hybrid Model"):
    y_true = []
    y_pred = []
    target_labels = ["B-PERSON", "I-PERSON", "B-ADDRESS", "I-ADDRESS", "B-EMAIL", "I-EMAIL"]

    model.eval()
    with torch.no_grad():
        for text, annotations in test_data:
            # Tokenize the text with offsets
            tokenized_inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_offsets_mapping=True,
                return_tensors="pt",
                return_special_tokens_mask=True
            )
            offset_mapping = tokenized_inputs.pop('offset_mapping')[0].tolist()
            attention_mask = tokenized_inputs['attention_mask'].to(device)
            input_ids = tokenized_inputs['input_ids'].to(device)
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            predictions = model(input_ids, attention_mask)
            pred_labels = [LABEL_MAP_REV[p] for p in predictions[0]]
            # Prepare true labels
            true_labels = ["O"] * len(word_ids)
            entities = [{'start': start, 'end': end, 'label': label} for start, end, label in annotations]
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    # Special tokens
                    true_labels[idx] = 'PAD'
                else:
                    token_start, token_end = offset_mapping[idx]
                    token_label = 'O'
                    for entity in entities:
                        if token_end <= entity['start']:
                            continue
                        if token_start >= entity['end']:
                            continue
                        if token_start == entity['start']:
                            token_label = 'B-' + entity['label']
                        else:
                            token_label = 'I-' + entity['label']
                        break  # Once labeled, break
                    true_labels[idx] = token_label
            # Only consider labels where attention_mask is 1 (excluding padding)
            for idx in range(len(attention_mask[0])):
                if attention_mask[0][idx] == 1:
                    y_true.append(true_labels[idx])
                    y_pred.append(pred_labels[idx])

    # Generate classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, labels=target_labels, zero_division=0))

# Function to evaluate spaCy model
def evaluate_spacy_model(nlp_model, test_data, model_name="Model"):
    true_entities = []
    pred_entities = []
    target_labels = ["PERSON", "ADDRESS", "EMAIL"]
    
    # Map spaCy labels to our target labels
    SPACY_LABEL_MAP = {
        'PERSON': 'PERSON',
        'GPE': 'ADDRESS',
        'LOC': 'ADDRESS',
        'FAC': 'ADDRESS',
        'ORG': 'ADDRESS',  # Organizations can be part of addresses
    }
    EMAIL_REGEX = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    
    for text, annotations in test_data:
        doc = nlp_model(text)
        
        # True entities with character offsets
        for start_char, end_char, label in annotations:
            if label in target_labels:
                true_entities.append((start_char, end_char, label))
        
        # Predicted entities from spaCy, mapped to target labels
        for ent in doc.ents:
            if ent.label_ in SPACY_LABEL_MAP:
                mapped_label = SPACY_LABEL_MAP[ent.label_]
                pred_entities.append((ent.start_char, ent.end_char, mapped_label))
        
        # Extract EMAIL entities using regex
        for match in EMAIL_REGEX.finditer(text):
            pred_entities.append((match.start(), match.end(), 'EMAIL'))
    
    # Initialize counts
    tp_counts = {label: 0 for label in target_labels}
    fp_counts = {label: 0 for label in target_labels}
    fn_counts = {label: 0 for label in target_labels}
    
    # Matching predicted entities to true entities
    matched_pred_indices = set()
    for i, (true_start, true_end, true_label) in enumerate(true_entities):
        matched = False
        for j, (pred_start, pred_end, pred_label) in enumerate(pred_entities):
            if pred_label == true_label and max(true_start, pred_start) < min(true_end, pred_end):
                matched = True
                matched_pred_indices.add(j)
                tp_counts[true_label] += 1
                break
        if not matched:
            fn_counts[true_label] += 1
    
    # Any predicted entities not matched are false positives
    for j, (pred_start, pred_end, pred_label) in enumerate(pred_entities):
        if j not in matched_pred_indices:
            fp_counts[pred_label] += 1
    
    # Compute metrics
    metrics = {}
    for label in target_labels:
        tp = tp_counts[label]
        fp = fp_counts[label]
        fn = fn_counts[label]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        support = tp + fn
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    # Compute overall metrics
    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0.0
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1-score'] for m in metrics.values()) / len(metrics)
    
    # Display the classification report
    print(f"\nClassification Report for {model_name}:")
    print("{:<12} {:>9} {:>9} {:>9} {:>9}".format('Label', 'Precision', 'Recall', 'F1-score', 'Support'))
    for label in target_labels:
        m = metrics[label]
        print("{:<12} {:9.2f} {:9.2f} {:9.2f} {:9d}".format(
            label, m['precision']*100, m['recall']*100, m['f1-score']*100, m['support']))
    print("{:<12} {:9.2f} {:9.2f} {:9.2f}".format(
        'Micro avg', micro_precision*100, micro_recall*100, micro_f1*100))
    print("{:<12} {:9.2f} {:9.2f} {:9.2f}".format(
        'Macro avg', macro_precision*100, macro_recall*100, macro_f1*100))

# Re-evaluate the hybrid model with corrected test data
evaluate_hybrid_model(model, tokenizer, test_data, model_name="Hybrid Model")

# Evaluate custom spaCy model (if available)
try:
    nlp_custom = spacy.load("custom_ner_model")
    evaluate_spacy_model(nlp_custom, test_data, model_name="Custom spaCy Model")
except Exception as e:
    print("Custom spaCy model not found or failed to load:", e)

# Evaluate original pre-trained spaCy model
nlp_original = spacy.load('en_core_web_sm')  # Load original pre-trained model
evaluate_spacy_model(nlp_original, test_data, model_name="Original spaCy Model")

# Display sample paragraphs and model predictions for comparison
for idx, (text, annotations) in enumerate(test_data[:5], 1):
    doc_original = nlp_original(text)
    # Load custom spaCy model if available
    try:
        doc_custom = nlp_custom(text)
    except:
        doc_custom = None

    # Prepare input for the hybrid model
    tokenized_inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
        return_tensors="pt",
        return_special_tokens_mask=True
    )
    offset_mapping = tokenized_inputs.pop('offset_mapping')[0].tolist()
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    input_ids = tokenized_inputs['input_ids'].to(device)
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    predictions = model(input_ids, attention_mask)
    pred_labels = [LABEL_MAP_REV[p] for p in predictions[0]]

    # Extract predicted entities from the hybrid model
    hybrid_entities = []
    current_entity_tokens = []
    current_label = None
    for idx_token, label in enumerate(pred_labels):
        if attention_mask[0][idx_token] == 0:
            continue
        if label != "O" and label != "PAD":
            entity_label = label.split('-')[1]
            start_char, end_char = offset_mapping[idx_token]
            token_text = text[start_char:end_char].strip()
            if current_label == entity_label:
                # Continue the current entity
                current_entity_tokens.append(token_text)
            else:
                # Save the previous entity if exists
                if current_entity_tokens:
                    entity_text = ' '.join(current_entity_tokens)
                    hybrid_entities.append((current_label, entity_text))
                # Start a new entity
                current_label = entity_label
                current_entity_tokens = [token_text]
        else:
            if current_entity_tokens:
                # Save the current entity
                entity_text = ' '.join(current_entity_tokens)
                hybrid_entities.append((current_label, entity_text))
                current_entity_tokens = []
                current_label = None
    # Save the last entity if exists
    if current_entity_tokens:
        entity_text = ' '.join(current_entity_tokens)
        hybrid_entities.append((current_label, entity_text))

    print(f"\nParagraph {idx}:\n{text}\n")

    if doc_custom:
        print("Custom spaCy Model Detected Entities:")
        for ent in doc_custom.ents:
            print(f"- '{ent.text}' : {ent.label_}")
    else:
        print("Custom spaCy Model not available.")
    
    print("\nOriginal spaCy Model Detected Entities:")
    for ent in doc_original.ents:
        print(f"- '{ent.text}' : {ent.label_}")
    
    print("\nHybrid Model Detected Entities:")
    for label, entity in hybrid_entities:
        print(f"- '{entity}' : {label}")
    print("-" * 80)
