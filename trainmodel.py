from ultralytics import YOLO
import argparse
import os

def train_model(model_config, data_config, epochs, batch_size, img_size, project_dir, device):
    """
    Train a YOLOv8-CSP model.

    Args:
        model_config (str): Path to the YOLO model configuration file (e.g., 'yolov8-csp.yaml').
        data_config (str): Path to the dataset configuration file (e.g., 'data.yaml').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Input image size for training.
        project_dir (str): Directory to save training results.
        device (str): Device to use for training ('cpu' or 'cuda').
    """
    # Initialize YOLO model
    model = YOLO(model_config)

    # Start training
    model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project_dir,
        device=device,
    )

    print(f"Training complete. Results saved in: {project_dir}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a YOLOv8-CSP model on PSL hand gestures.")
    parser.add_argument("--model_config", type=str, default="yolov8-csp.yaml", help="Path to model config file.")
    parser.add_argument("--data_config", type=str, default="data.yaml", help="Path to dataset config file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size for training.")
    parser.add_argument("--project_dir", type=str, default="runs/train", help="Directory to save training results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cpu' or 'cuda').")
    args = parser.parse_args()

    # Train the model
    if not os.path.exists(args.data_config):
        raise FileNotFoundError(f"Dataset config file not found: {args.data_config}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config file not found: {args.model_config}")

    train_model(
        model_config=args.model_config,
        data_config=args.data_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        project_dir=args.project_dir,
        device=args.device,
    )
