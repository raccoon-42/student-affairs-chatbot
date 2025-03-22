from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "distilbert-base-uncased"

# Download model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Save model locally
model.save_pretrained("./distilbert_offline")
tokenizer.save_pretrained("./distilbert_offline")