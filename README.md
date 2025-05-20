import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import random
import logging
from pathlib import Path
from dataset_utils import load_ucmerced_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class UCMercedCaptioningDataset(Dataset):
    def __init__(self, image_paths, captions, processor, tokenizer, max_length=64):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            
            caption = self.captions[idx]

            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            
            # Process text
            labels = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()

            return {
                "pixel_values": pixel_values,
                "labels": labels
            }
        except Exception as e:
            logging.error(f"Error loading item {idx}: {str(e)}")
            raise

def main():
    try:
        # Create output directory
        output_dir = Path("SCRIPTS/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device and random seed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Set deterministic training for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Initialize model and processors
        logging.info("Initializing model and processors...")
        model_name = "google/vit-base-patch16-224-in21k"
        
        # Initialize and configure tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Use smaller model
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k",
            "distilgpt2"  # Smaller model
        )

        # Configure model
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.max_length = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 2
        
        # Memory optimizations
        model.config.use_cache = False
        
        model.to(device)

        # Load data
        image_paths, captions = load_ucmerced_data()
        
        # Split data
        train_idx = int(0.8 * len(image_paths))
        train_image_paths = image_paths[:train_idx]
        train_captions = captions[:train_idx]
        val_image_paths = image_paths[train_idx:]
        val_captions = captions[train_idx:]

        logging.info(f"Training samples: {len(train_image_paths)}")
        logging.info(f"Validation samples: {len(val_image_paths)}")

        # Create datasets
        train_dataset = UCMercedCaptioningDataset(
            train_image_paths, train_captions, image_processor, tokenizer
        )
        val_dataset = UCMercedCaptioningDataset(
            val_image_paths, val_captions, image_processor, tokenizer
        )

        # Smaller batch size for memory efficiency
        batch_size = 2
        logging.info(f"Using batch size: {batch_size}")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Training parameters
        num_epochs = 10
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Add gradient clipping
        max_grad_norm = 1.0
        
        # Early stopping parameters
        patience = 3
        min_delta = 0.01
        patience_counter = 0
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            model.train()
            train_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

                # Clear memory
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_dataloader)
            logging.info(f"Training loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()
                    
                    # Clear memory
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            avg_val_loss = val_loss / len(val_dataloader)
            logging.info(f"Validation loss: {avg_val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_dir = output_dir / "best_model"
                model_save_dir.mkdir(exist_ok=True)
                model.save_pretrained(model_save_dir)
                tokenizer.save_pretrained(model_save_dir)
                torch.save(checkpoint, model_save_dir / "checkpoint.pt")
                logging.info("Saved best model!")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save latest checkpoint
            torch.save(checkpoint, output_dir / "latest_checkpoint.pt")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
