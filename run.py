import argparse
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import imageio
from torch.optim import Adam
import models
from datasets.image_folder import PairedImageFolders
import torch.nn.functional as F

def setup_logging(log_path):
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def train(args):
    """Handles the training process."""
    logging.info("Starting training...")

    # Dataset and DataLoader
    train_dataset = PairedImageFolders(
        os.path.join(args.data_root, 'Image_train'),
        os.path.join(args.data_root, 'GT_train'),
        size=args.image_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    encoder_mode = {
        'name': 'evp',
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'out_chans': 256,
        'qkv_bias': True,
        'norm_layer': nn.LayerNorm,
        'act_layer': nn.GELU,
        'use_rel_pos': True,
        'rel_pos_zero_init': True,
        'window_size': 14,
        'global_attn_indexes': [2, 5, 8, 11],
        'prompt_embed_dim': 256,
    }

    model_spec = {
        'name': 'sam',
        'args': {
            'inp_size': args.image_size,
            'encoder_mode': encoder_mode,
            'loss': 'bce'
        }
    }
    model = models.make(model_spec).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for img, gt in pbar:
            img, gt = img.to(device), gt.to(device)

            model.set_input(img, gt)
            model.optimize_parameters()

            epoch_loss += model.loss_G.item()
            pbar.set_postfix(loss=model.loss_G.item())

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), args.checkpoint_path)
        if os.path.exists(args.checkpoint_path):
            logging.info(f"Checkpoint saved to {args.checkpoint_path}")
        else:
            logging.error(f"Failed to save checkpoint to {args.checkpoint_path}")

class TestDataset(Dataset):
    """Custom dataset for test images."""
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.filenames = sorted(os.listdir(path))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        image = self.transform(image)
        return image, original_size, self.filenames[idx]

def test(args):
    """Handles the testing process."""
    logging.info("Starting testing...")

    # Dataset and DataLoader
    test_dataset = TestDataset(
        os.path.join(args.data_root, 'Image_test'),
        size=args.image_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model
    encoder_mode = {
        'name': 'evp',
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'out_chans': 256,
        'qkv_bias': True,
        'norm_layer': nn.LayerNorm,
        'act_layer': nn.GELU,
        'use_rel_pos': True,
        'rel_pos_zero_init': True,
        'window_size': 14,
        'global_attn_indexes': [2, 5, 8, 11],
        'prompt_embed_dim': 256,
    }

    model_spec = {
        'name': 'sam',
        'args': {
            'inp_size': args.image_size,
            'encoder_mode': encoder_mode,
            'loss': 'bce'
        }
    }
    model = models.make(model_spec).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint not found at {args.checkpoint_path}")
        return

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded from checkpoint.")

    # Testing loop
    pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for img, original_size, filename in pbar:
            img = img.to(device)

            pred_mask = model.infer(img)
            pred_mask = torch.sigmoid(pred_mask)

            w, h = original_size[0].item(), original_size[1].item()

            resized_mask = F.interpolate(pred_mask, size=(h, w), mode='bilinear', align_corners=False)

            resized_mask = resized_mask.squeeze().cpu().numpy()
            resized_mask = (resized_mask * 255).astype(np.uint8)

            base_filename = os.path.splitext(filename[0])[0]
            output_filename = f"{base_filename}.png"
            output_path = os.path.join(args.output_dir, 'predictions', output_filename)

            imageio.imwrite(output_path, resized_mask)
            pbar.set_postfix(saved=output_filename)

    logging.info(f"Testing complete. Predictions saved to {os.path.join(args.output_dir, 'predictions')}")

def main():
    """Main function to run the training and testing."""
    parser = argparse.ArgumentParser(description="Train and test the model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Mode: 'train' or 'test'")
    parser.add_argument('--data_root', type=str, default='../datasets/NEU-RSDDS-AUG', help="Dataset root directory")
    parser.add_argument('--checkpoint_path', type=str, default='./output/checkpoint.pth', help="Path to save/load checkpoint")
    parser.add_argument('--output_dir', type=str, default='./output', help="Output directory")
    parser.add_argument('--log_file', type=str, default='./output/result.log', help="Log file path")

    # Hyperparameters from the original project
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--image_size', type=int, default=384, help="Image size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)

    setup_logging(args.log_file)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__ == '__main__':
    main()