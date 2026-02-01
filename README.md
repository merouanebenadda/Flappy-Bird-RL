# Deep Q-Learning to play Flappy Bird

A Deep Q-Network (DQN) implementation for training an agent to play Flappy Bird.

## Usage
### Training

Set test_mode = False in main.py to begin training. Models are saved periodically based on the ModelSaveFreq parameter.

### Testing

Set test_mode = True and specify the model file in test_model_path. This loads the weights and enables human-mode rendering for visualization.


```bash
# Run the program
python3 main.py