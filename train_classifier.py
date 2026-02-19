
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "classification_results_refined.csv"
MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

# Set Random Seed for Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class SimpleAugmenter:
    """Basic synonym replacement for aviation text to boost dataset size."""
    def __init__(self):
        self.synonyms = {
            "aircraft": ["airplane", "plane", "jet", "ship"],
            "pilot": ["captain", "first officer", "crew", "pic"],
            "reported": ["stated", "mentioned", "declared", "noted"],
            "approach": ["landing phase", "descent", "final"],
            "runway": ["strip", "tarmac", "rwy"],
            "collision": ["impact", "crash", "strike"],
            "conflict": ["incident", "issue", "near miss"],
            "atc": ["tower", "controller", "ground control"],
            "failure": ["malfunction", "issue", "problem"]
        }

    def augment(self, text):
        words = text.split()
        new_words = []
        changed = False
        for word in words:
            lower_word = word.lower().strip(".,")
            if lower_word in self.synonyms and random.random() > 0.5:
                replacement = random.choice(self.synonyms[lower_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
                changed = True
            else:
                new_words.append(word)
        
        return " ".join(new_words) if changed else text

class AviationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]


        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data():
    print("Loading classification results for pseudo-labeling...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Filter High Confidence Rows (The "Teacher" Labels)
    # We trust 'Rule' completely, and 'Embedding' > 0.40
    # We REJECT 'UNK' and low confidence rows
    mask = (df['method'] == 'Rule') | ( (df['method'] == 'Embedding') & (df['confidence'] >= 0.45) ) 
    # Note: Using 0.45 just to be extra safe for training data quality
    
    clean_df = df[mask].copy()
    print(f"Filtered {len(clean_df)} high-confidence samples from {len(df)} total.")
    
    # 2. Label Encoding
    label_map = {label: idx for idx, label in enumerate(clean_df['predicted_code'].unique())}
    inverse_map = {idx: label for label, idx in label_map.items()}
    print(f"Found {len(label_map)} unique categories: {list(label_map.keys())}")
    
    clean_df['label_idx'] = clean_df['predicted_code'].map(label_map)
    
    # 3. Augmentation
    print("Applying Data Augmentation...")
    augmenter = SimpleAugmenter()
    
    augmented_texts = []
    augmented_labels = []
    
    for _, row in clean_df.iterrows():
        # Add original
        augmented_texts.append(row['event_string'])
        augmented_labels.append(row['label_idx'])
        
        # Add 2 Aggmented versions
        for _ in range(2):
            aug_text = augmenter.augment(row['event_string'])
            if aug_text != row['event_string']:
                augmented_texts.append(aug_text)
                augmented_labels.append(row['label_idx'])
                
    print(f"Dataset expanded to {len(augmented_texts)} samples after augmentation.")
    
    return augmented_texts, augmented_labels, label_map

def train_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    texts, labels, label_map = load_and_prepare_data()
    num_classes = len(label_map)
    

    # Split
    # Removed stratify to avoid errors with classes that have < 2 samples after augmentation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )
    
    # Tokenizer
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = AviationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = AviationDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    print(f"Loading {MODEL_NAME} model...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Optimizer

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Training Loop
    print("\nStarting Training...")
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_accuracy = []
        val_preds = []
        val_true = []
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            
        avg_val_acc = np.mean(val_accuracy)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "best_model_roberta.pt")
            print("--> Best model saved.")

    print("\nTraining Complete.")
    print(f"Final Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
