# Training

The files within this folder define all logic related to training and validating the model.

To run training (**Note:** This should be done in a folder separate from this repository):
1. Ensure poke-env is installed
```
pip install poke-env
```

2. Configure your Pokémon Showdown Server
```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

3. From the root of the repository, run the following command:
```
python -m pokemonnn.training.train
```