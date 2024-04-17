# README
Code repository of article `Multimodal tactile sensing fused with vision for dexterous robotic housekeeping`.

### 1. `Fabric_recognition.m`

#### Overview
This Matlab file returns the trained classifier and its accuracy. The code recreates a classification model trained in a classification learner. You can use the generated code to automatically train the same model on new data, or to learn how to train the model programmatically.Make sure to place the 'Fabric_dataset.mat' file in the specified directory path. 

### 2. `Tactile_recognition.m`

#### Overview
This Matlab file is used for tactile recognition.

### 3. `main.py`

#### Overview
This Python file is used to complete desktop-cleaning task, including object location, stable grasping and object recognition, etc.

### 4. `strategy.py`

#### Overview
This Python file is used to determine the robot's grasping strategy with different objects.

### 5. `decodedata.py`

#### Overview
This Python file is used to decode the tactile signals.

### 6. `sercommunication.py`

#### Overview
This Python file is used for serial communication
