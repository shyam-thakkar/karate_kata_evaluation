# Karate Kata Evaluation System

This project is an automated Karate Kata evaluation system leveraging machine learning and deep learning techniques. The system uses Google's MoveNet Thunder for pose detection and a custom grading algorithm to score kata performances from 0 to 10.

## Features
- **Pose Detection:** Utilizes MoveNet Thunder for accurate keypoint extraction.
- **Deep Learning Model:** Classifies and evaluates karate kata performances using a Dense Neural Network (DNN).
- **Custom Grading Algorithm:** Scores performances on a scale from 0 to 10.
- **GUI Interface:** Provides a user-friendly interface for easy interaction and live predictions.

## Project Structure
- `evaluate_model.py`: Contains code for evaluating the trained model.
- `gui.py`: Implements the graphical user interface for the system.
- `keypoint_extraction.py`: Handles keypoint extraction using MoveNet.
- `making_csv_dataset.py`: Script to generate a dataset in CSV format from images.
- `train_model.py`: Code for training the DNN model.
- `videotoimage.py`: Converts video frames to images for dataset preparation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/shyam-thakkar/karate_kata_evaluation.git
    cd karate_kata_evaluation
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Extract keypoints from images:
    ```bash
    python keypoint_extraction.py
    ```
2. Train the model:
    ```bash
    python train_model.py
    ```
3. Evaluate the model:
    ```bash
    python evaluate_model.py
    ```
4. Launch the GUI:
    ```bash
    python gui.py
    ```

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Contact
For any questions or inquiries, please contact [Shyam Thakkar](https://github.com/shyam-thakkar).

