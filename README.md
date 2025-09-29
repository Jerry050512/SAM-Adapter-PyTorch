# Simplified SAM Model for Defect Segmentation

This project provides a streamlined implementation of the SAM (Segment Anything Model) for defect segmentation, specifically tailored for the NEU-RSSDDS-AUG dataset. The codebase has been simplified to focus on the core architecture and functionality, with a single script to handle both training and testing.

## 1. Environment Setup

To get started, you'll need to install the required Python libraries. It is recommended to use a virtual environment to avoid conflicts with other projects.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the dependencies
pip install -r requirements.txt
```

## 2. Data Preparation

This project is configured to work with the **NEU-RSSDDS-AUG** dataset. You need to structure your data as follows:

```
../datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/
    │   ├── 001.bmp
    │   ├── 002.bmp
    │   └── ...
    ├── GT_train/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── Image_test/
        ├── 001.bmp
        ├── 002.bmp
        └── ...
```

- **`Image_train`**: Contains the training images in `.bmp` format.
- **`GT_train`**: Contains the corresponding ground truth masks for training, in `.png` format.
- **`Image_test`**: Contains the test images in `.bmp` format.

Make sure the file names in `Image_train` and `GT_train` match.

## 3. Training

To train the model, run the `run.py` script in `train` mode. The script will use the data from the `NEU-RSDDS-AUG` directory, train the model, and save the checkpoint to `./output/checkpoint.pth`.

```bash
python run.py --mode train
```

You can customize the training parameters, such as the number of epochs and batch size, using command-line arguments:

```bash
# Example with custom parameters
python run.py --mode train --epochs 50 --batch_size 8 --lr 0.0002
```

The training progress and logs will be saved to `./output/result.log`.

## 4. Testing

To test the model, run the `run.py` script in `test` mode. The script will load the saved checkpoint from `./output/checkpoint.pth`, process the images in the `Image_test` directory, and save the predicted masks to `./output/predictions/`.

```bash
python run.py --mode test
```

The predicted masks will be saved as `.png` files with the same names as the original test images. The output masks will be resized to match the original image dimensions.

## 5. Output Files

All output files will be saved in the `output` directory:

- **`./output/checkpoint.pth`**: The model checkpoint saved during training.
- **`./output/result.log`**: The log file containing training and testing information.
- **`./output/predictions/`**: The directory where predicted masks are saved during testing.