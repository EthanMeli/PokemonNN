# Training

The files within this folder define all logic related to training and validating the model.

To run training (**Note:** This should be done in a folder separate from this repository):
1. Ensure poke-env is installed
```bash
pip install poke-env
```

2. Configure your Pokémon Showdown Server
```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
```

3. Run the following commands from the root of the repository to start up the showdown servers:
```bash
cd pokemonnn/training
python launch_showdown.py
```

3. In a separate terminal, from the root of the repository, run the following command:
```
python -m pokemonnn.training.train
```

**Note:** The default constants for training (i.e. TOTAL_UPDATES, BATTLES_PER_UPDATE, and VALIDATE_EVERY) are intentionally small here and should be updated for longer training runs. You can also mess around with the number of instances and battles per update by modifying the BATTLES_PER_UPDATE and N_INSTANCES in train.py and NUM_INSTANCES in launch_showdown.py. Make sure that N_INSTANCES and NUM_INSTANCES are the same between the two files.