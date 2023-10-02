# Words Spoken Daily

Welcome to the repository for the Words Spoken Daily project! This repository contains the inference code and pre-trained model for our speaker tracking system, as described in the paper titled "Words Spoken Daily among Individuals with Neurodegenerative Conditions: A Pilot Study."

### Participant Data Requirements
To use this code effectively, you'll need to provide the following for each participant:
- A dedicated folder containing the participant's recordings.
- A separate recording of reading passages to enroll the participant's voice.

### Download Our Pre-trained Model
To get started, please download our pre-trained speaker tracking model from this [Google Drive link](https://drive.google.com/file/d/1A-WV_qTYyj4MXyU8fKH1B05EH-3B3ISd/view?usp=sharing) and then place the downloaded model file into the `model` folder of this repository.

### Setting up the Environment

To set up the required Python environment for running the code, you can use the provided `requirements.txt` file. Here's how:

1. First, make sure you have Python 3 installed on your system.

2. Create a virtual environment (optional but recommended):

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages using pip:

   ```
   pip install -r requirements.txt
   ```

### Running the Code
Once you've obtained the necessary data and downloaded the pre-trained model, you can run the code using the following command:
```shell
python3 extract_target.py -e PATH_TO_ENROLLMENT_RECORDING -i PATH_TO_DAILY_RECORDING_FOLDER -o OUTPUT_FOLDER -t THRESHOLD
```
Replace `PATH_TO_ENROLLMENT_RECORDING`, `PATH_TO_DAILY_RECORDING_FOLDER`, `OUTPUT_FOLDER`, and `THRESHOLD` with the appropriate file paths and desired threshold value.

Feel free to reach out if you have any questions or need further assistance with our code. 