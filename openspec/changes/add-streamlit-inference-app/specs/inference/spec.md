## ADDED Requirements
### Requirement: Inference Web App (Streamlit)
The system SHALL provide a small web application that accepts an image upload and returns model predictions (probabilities) for the three myna classes.

#### Scenario: Upload image and receive predictions
- **WHEN** a user uploads a valid JPG/PNG image via the app
- **THEN** the app SHALL display the uploaded image
- **AND** the app SHALL display the model's predicted probabilities for each class sorted by confidence

### Requirement: Exported Model Artifact
The system SHALL provide a reproducible training script that exports a saved Keras model and a companion labels file (JSON) describing the class order.

#### Scenario: Train and export model
- **WHEN** an engineer runs the training script with a path to a folder of labeled subfolders
- **THEN** the script SHALL save a `models/<name>/` saved_model directory
- **AND** the script SHALL save `models/labels.json` containing the class names in index order
