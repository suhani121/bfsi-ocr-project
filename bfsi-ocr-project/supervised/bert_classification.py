import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess dataset
df = pd.read_csv(r"bfsi-ocr-project\bfsi-ocr-project\supervised\transaction_dataset.csv")

# Encode categories
label_encoder = LabelEncoder()
df["Category"] = label_encoder.fit_transform(df["Category"])
categories = list(label_encoder.classes_)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Description"], df["Category"], test_size=0.2, random_state=42
)

# Tokenize text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=64)

# Create custom dataset class
class TransactionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TransactionDataset(train_encodings, train_labels.tolist())
val_dataset = TransactionDataset(val_encodings, val_labels.tolist())

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="epoch",
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Define preprocessing function for new descriptions
def preprocess_description(description):
    description = re.sub(r"UPI/\d+/", "", description)  # Remove UPI IDs
    description = re.sub(r"\d+", "", description)  # Remove numeric codes
    return description.strip()

# Load new dataset for classification
data_file = r"Supervised/extracted_data.csv"
df = pd.read_csv(data_file)

# Preprocess descriptions
df['cleaned_desc'] = df['desc'].dropna().apply(preprocess_description)
descriptions = df['cleaned_desc'].tolist()

# Define new categories
categories = [
    "Healthcare", "Education", "Personal Development", 
    "Savings and Investments", "Debt Payments", 
    "Miscellaneous or Other", "Food and Dining",
    "Housing", "Transportation", "Clothing and Accessories", 
    "Entertainment and Recreation"
]

# Load tokenizer and trained model
model_path = "./trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Classify new descriptions
batch_size = 32
predicted_categories = []

model.eval()
with torch.no_grad():
    for i in range(0, len(descriptions), batch_size):
        batch_descriptions = descriptions[i:i+batch_size]
        tokenized_batch = tokenizer(
            batch_descriptions, 
            truncation=True, 
            padding=True, 
            max_length=64, 
            return_tensors="pt"
        )
        
        outputs = model(**tokenized_batch)
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()

        for pred in predictions:
            if pred < len(categories):
                predicted_categories.append(categories[pred])
            else:
                predicted_categories.append("Unknown")

# Add predictions to DataFrame
df['predicted_category'] = None
df.loc[df['desc'].notna(), 'predicted_category'] = predicted_categories

# Save classified data
output_file = "classified_data.csv"
df.to_csv(output_file, index=False)

print(f"Classified data saved to: {output_file}")
