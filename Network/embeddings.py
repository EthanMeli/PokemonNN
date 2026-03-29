"""
embeddings.py
======================
Generate LLM-initialized entity embeddings for all Gen 1 Pokémon species and moves.

Architecture context (from project outline, Section 4.1):
  - Every entity gets a frozen 768-dim embedding from a pretrained sentence encoder
    (Sentence-BERT) applied to a rich text description.
  - A learned linear projection (768 → 64) maps these into the RL latent space
    at training time — that projection is NOT part of this script.
  - This script produces the raw frozen embeddings that the model loads.

Gen 1 scope:
  - 151 species (no abilities in Gen 1)
  - ~165 moves
  - Items are not used in gen1randombattle format
  - We also generate an "unknown" embedding for unrevealed Pokémon

Usage:
  pip install sentence-transformers torch
  python generate_embeddings.py [--model all-mpnet-base-v2] [--output-dir embeddings/]

Outputs:
  embeddings/
    species_embeddings.pt    — dict: {species_name: tensor(768,)}
    move_embeddings.pt       — dict: {move_name: tensor(768,)}
    unknown_embedding.pt     — tensor(768,) for unrevealed Pokémon
    embedding_index.pt       — dict with ordered lists + name→index mappings
    metadata.json            — generation metadata (model, date, counts)
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Gen 1 Entity Data
# ---------------------------------------------------------------------------
# Comprehensive Gen 1 species data: name, types, base stats, and a short
# flavour/strategy description drawn from common competitive knowledge.
# Base stats are from the original Gen 1 games (Red/Blue/Yellow).
# ---------------------------------------------------------------------------

GEN1_SPECIES: List[Dict] = [
  # --- Starters & evolutions ---
  {"name": "bulbasaur", "types": ["grass", "poison"], "hp": 45, "atk": 49, "def": 49, "spa": 65, "spd": 65, "spe": 45,
    "desc": "A small Grass/Poison type. Bulbasaur can learn status moves like Sleep Powder and Leech Seed. It evolves into Ivysaur."},
  {"name": "ivysaur", "types": ["grass", "poison"], "hp": 60, "atk": 62, "def": 63, "spa": 80, "spd": 80, "spe": 60,
    "desc": "The evolved form of Bulbasaur. Ivysaur has improved bulk and special stats, with access to Razor Leaf and Sleep Powder."},
  {"name": "venusaur", "types": ["grass", "poison"], "hp": 80, "atk": 82, "def": 83, "spa": 100, "spd": 100, "spe": 80,
    "desc": "A powerful Grass/Poison type with high Special. Venusaur excels as a tank with Sleep Powder, Razor Leaf, and Sludge Bomb. It can wall many Water and Electric types."},
  {"name": "charmander", "types": ["fire"], "hp": 39, "atk": 52, "def": 43, "spa": 60, "spd": 50, "spe": 65,
    "desc": "A small Fire type. Charmander is fast but fragile, with access to Fire-type moves and Slash."},
  {"name": "charmeleon", "types": ["fire"], "hp": 58, "atk": 64, "def": 58, "spa": 80, "spd": 65, "spe": 80,
    "desc": "The evolved form of Charmander. Charmeleon has improved speed and Special for stronger Fire-type attacks."},
  {"name": "charizard", "types": ["fire", "flying"], "hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100,
    "desc": "A Fire/Flying type with high Special and Speed. Charizard is a potent special attacker with Fire Blast and can hit many types super-effectively. Weak to Rock, Water, and Electric."},
  {"name": "squirtle", "types": ["water"], "hp": 44, "atk": 48, "def": 65, "spa": 50, "spd": 64, "spe": 43,
    "desc": "A small Water type with decent defense. Squirtle can learn Surf, Ice Beam, and defensive moves."},
  {"name": "wartortle", "types": ["water"], "hp": 59, "atk": 63, "def": 80, "spa": 65, "spd": 80, "spe": 58,
    "desc": "The evolved form of Squirtle. Wartortle has solid bulk and access to Water and Ice coverage moves."},
  {"name": "blastoise", "types": ["water"], "hp": 79, "atk": 83, "def": 100, "spa": 85, "spd": 105, "spe": 78,
    "desc": "A bulky Water type with excellent Defense. Blastoise can use Surf, Ice Beam, and Earthquake. A reliable tank that walls many physical attackers."},
  # --- Bug lines ---
  {"name": "caterpie", "types": ["bug"], "hp": 45, "atk": 30, "def": 35, "spa": 20, "spd": 20, "spe": 45,
    "desc": "A weak Bug type. Caterpie can only use Tackle and String Shot. It evolves very early."},
  {"name": "metapod", "types": ["bug"], "hp": 50, "atk": 20, "def": 55, "spa": 25, "spd": 25, "spe": 30,
    "desc": "A cocoon Bug type. Metapod primarily uses Harden and has very limited offensive options."},
  {"name": "butterfree", "types": ["bug", "flying"], "hp": 60, "atk": 45, "def": 50, "spa": 80, "spd": 80, "spe": 70,
    "desc": "A Bug/Flying type with decent Special. Butterfree's main strength is Sleep Powder and Psychic. It can also use Stun Spore for paralysis."},
  {"name": "weedle", "types": ["bug", "poison"], "hp": 40, "atk": 35, "def": 30, "spa": 20, "spd": 20, "spe": 50,
    "desc": "A weak Bug/Poison type. Weedle has Poison Sting and String Shot. It evolves very early."},
  {"name": "kakuna", "types": ["bug", "poison"], "hp": 45, "atk": 25, "def": 50, "spa": 25, "spd": 25, "spe": 35,
    "desc": "A cocoon Bug/Poison type. Kakuna uses Harden and has very limited moves."},
  {"name": "beedrill", "types": ["bug", "poison"], "hp": 65, "atk": 80, "def": 40, "spa": 45, "spd": 80, "spe": 75,
    "desc": "A Bug/Poison type physical attacker. Beedrill has Twineedle and Swords Dance but is frail with a low Special stat."},
  {"name": "pidgey", "types": ["normal", "flying"], "hp": 40, "atk": 45, "def": 40, "spa": 35, "spd": 35, "spe": 56,
    "desc": "A common Normal/Flying type. Pidgey is weak but fast for an early-game Pokémon."},
  {"name": "pidgeotto", "types": ["normal", "flying"], "hp": 63, "atk": 60, "def": 55, "spa": 50, "spd": 50, "spe": 71,
    "desc": "The evolved form of Pidgey. Pidgeotto has decent Speed and can use Wing Attack and Quick Attack."},
  {"name": "pidgeot", "types": ["normal", "flying"], "hp": 83, "atk": 80, "def": 75, "spa": 70, "spd": 70, "spe": 91,
    "desc": "A fast Normal/Flying type. Pidgeot has good Speed and can use Hyper Beam, Wing Attack, and Mirror Move. Outclassed by other Normal types in Gen 1."},
  {"name": "rattata", "types": ["normal"], "hp": 30, "atk": 56, "def": 35, "spa": 25, "spd": 35, "spe": 72,
    "desc": "A fast but very frail Normal type. Rattata has Super Fang which always halves the target's HP, and Hyper Fang."},
  {"name": "raticate", "types": ["normal"], "hp": 55, "atk": 81, "def": 60, "spa": 50, "spd": 70, "spe": 97,
    "desc": "A fast Normal type. Raticate has Super Fang to halve HP and Hyper Fang for damage. Decent speed but frail."},
  {"name": "spearow", "types": ["normal", "flying"], "hp": 40, "atk": 60, "def": 30, "spa": 31, "spd": 31, "spe": 70,
    "desc": "An aggressive Normal/Flying type. Spearow hits harder than Pidgey but is frailer."},
  {"name": "fearow", "types": ["normal", "flying"], "hp": 65, "atk": 90, "def": 65, "spa": 61, "spd": 61, "spe": 100,
    "desc": "A fast and hard-hitting Normal/Flying type. Fearow has Drill Peck and Hyper Beam with base 100 Speed. A solid physical attacker."},
  {"name": "ekans", "types": ["poison"], "hp": 35, "atk": 60, "def": 44, "spa": 40, "spd": 54, "spe": 55,
    "desc": "A Poison type snake. Ekans can use Glare to paralyze and Wrap to trap opponents."},
  {"name": "arbok", "types": ["poison"], "hp": 60, "atk": 85, "def": 69, "spa": 65, "spd": 79, "spe": 80,
    "desc": "A Poison type with decent Attack. Arbok uses Glare for paralysis, Wrap for trapping, and Earthquake for coverage."},
  {"name": "pikachu", "types": ["electric"], "hp": 35, "atk": 55, "def": 30, "spa": 50, "spd": 40, "spe": 90,
    "desc": "A fast but frail Electric type. Pikachu has Thunderbolt and Thunder Wave. Very low bulk makes it a glass cannon."},
  {"name": "raichu", "types": ["electric"], "hp": 60, "atk": 90, "def": 55, "spa": 90, "spd": 80, "spe": 100,
    "desc": "A fast Electric type with good Special and Attack. Raichu has Thunderbolt, Thunder Wave, and Surf (via special event). Outclassed by Zapdos and Jolteon."},
  {"name": "sandshrew", "types": ["ground"], "hp": 50, "atk": 75, "def": 85, "spa": 20, "spd": 30, "spe": 40,
    "desc": "A defensive Ground type. Sandshrew has high Defense but is very slow with a terrible Special stat."},
  {"name": "sandslash", "types": ["ground"], "hp": 75, "atk": 100, "def": 110, "spa": 45, "spd": 55, "spe": 65,
    "desc": "A physically bulky Ground type. Sandslash has Earthquake and Swords Dance with excellent Defense but poor Special and Speed."},
  {"name": "nidoranf", "types": ["poison"], "hp": 55, "atk": 47, "def": 52, "spa": 40, "spd": 40, "spe": 41,
    "desc": "A small Poison type. Nidoran Female evolves into Nidorina and eventually the powerful Nidoqueen."},
  {"name": "nidorina", "types": ["poison"], "hp": 70, "atk": 62, "def": 67, "spa": 55, "spd": 55, "spe": 56,
    "desc": "The evolved form of Nidoran Female. Nidorina has decent all-around stats and evolves into Nidoqueen."},
  {"name": "nidoqueen", "types": ["poison", "ground"], "hp": 90, "atk": 82, "def": 87, "spa": 75, "spd": 85, "spe": 76,
    "desc": "A bulky Poison/Ground type. Nidoqueen has Earthquake, Thunderbolt, Ice Beam, and good mixed bulk. A versatile attacker and pivot."},
  {"name": "nidoranm", "types": ["poison"], "hp": 46, "atk": 57, "def": 40, "spa": 40, "spd": 40, "spe": 50,
    "desc": "A small Poison type. Nidoran Male evolves into Nidorino and eventually the powerful Nidoking."},
  {"name": "nidorino", "types": ["poison"], "hp": 61, "atk": 72, "def": 57, "spa": 55, "spd": 55, "spe": 65,
    "desc": "The evolved form of Nidoran Male. Nidorino is more offensive than Nidorina and evolves into Nidoking."},
  {"name": "nidoking", "types": ["poison", "ground"], "hp": 81, "atk": 92, "def": 77, "spa": 85, "spd": 75, "spe": 85,
    "desc": "An offensive Poison/Ground type. Nidoking has Earthquake, Thunderbolt, Ice Beam, and great coverage. Faster and more offensive than Nidoqueen."},
  {"name": "clefairy", "types": ["normal"], "hp": 70, "atk": 45, "def": 48, "spa": 60, "spd": 65, "spe": 35,
    "desc": "A Normal type with decent HP and Special. Clefairy can learn many TMs including Thunderbolt, Ice Beam, and Thunder Wave."},
  {"name": "clefable", "types": ["normal"], "hp": 95, "atk": 70, "def": 73, "spa": 85, "spd": 90, "spe": 60,
    "desc": "A bulky Normal type with great movepool coverage. Clefable can use Thunderbolt, Ice Beam, Thunder Wave, and has solid HP and Special stats."},
  {"name": "vulpix", "types": ["fire"], "hp": 38, "atk": 41, "def": 40, "spa": 50, "spd": 65, "spe": 65,
    "desc": "A Fire type fox. Vulpix has Flamethrower and Confuse Ray but low overall stats."},
  {"name": "ninetales", "types": ["fire"], "hp": 73, "atk": 76, "def": 75, "spa": 81, "spd": 100, "spe": 100,
    "desc": "A fast Fire type with good Special. Ninetales has Fire Blast, Confuse Ray, and decent speed. Outclassed by other Fire types but still usable."},
  {"name": "jigglypuff", "types": ["normal"], "hp": 115, "atk": 45, "def": 20, "spa": 25, "spd": 25, "spe": 20,
    "desc": "A Normal type with very high HP but terrible defenses. Jigglypuff can use Sing to put foes to sleep."},
  {"name": "wigglytuff", "types": ["normal"], "hp": 140, "atk": 70, "def": 45, "spa": 50, "spd": 50, "spe": 45,
    "desc": "A Normal type with enormous HP but poor defenses and speed. Wigglytuff can use Thunderbolt, Ice Beam, and has a wide movepool."},
  {"name": "zubat", "types": ["poison", "flying"], "hp": 40, "atk": 45, "def": 35, "spa": 30, "spd": 40, "spe": 55,
    "desc": "A Poison/Flying type bat. Zubat is weak but can use Confuse Ray and Leech Life."},
  {"name": "golbat", "types": ["poison", "flying"], "hp": 75, "atk": 80, "def": 70, "spa": 65, "spd": 75, "spe": 90,
    "desc": "A fast Poison/Flying type. Golbat has Confuse Ray and decent speed but limited offensive moves in Gen 1."},
  {"name": "oddish", "types": ["grass", "poison"], "hp": 45, "atk": 50, "def": 55, "spa": 75, "spd": 65, "spe": 30,
    "desc": "A Grass/Poison type. Oddish can use Sleep Powder and Absorb. It evolves into Gloom."},
  {"name": "gloom", "types": ["grass", "poison"], "hp": 60, "atk": 65, "def": 70, "spa": 85, "spd": 75, "spe": 40,
    "desc": "The evolved form of Oddish. Gloom has decent Special and access to Sleep Powder and Mega Drain."},
  {"name": "vileplume", "types": ["grass", "poison"], "hp": 75, "atk": 80, "def": 85, "spa": 100, "spd": 90, "spe": 50,
    "desc": "A bulky Grass/Poison type with high Special. Vileplume uses Sleep Powder, Mega Drain, and Sludge Bomb. Slow but powerful."},
  {"name": "paras", "types": ["bug", "grass"], "hp": 35, "atk": 70, "def": 55, "spa": 45, "spd": 55, "spe": 25,
    "desc": "A Bug/Grass type with Spore, the most accurate sleep move. Paras is very slow and has many weaknesses, especially to Fire."},
  {"name": "parasect", "types": ["bug", "grass"], "hp": 60, "atk": 95, "def": 80, "spa": 60, "spd": 80, "spe": 30,
    "desc": "A Bug/Grass type with access to Spore. Parasect has decent Attack but is extremely slow and has a crippling 4x Fire weakness."},
  {"name": "venonat", "types": ["bug", "poison"], "hp": 60, "atk": 55, "def": 50, "spa": 40, "spd": 55, "spe": 45,
    "desc": "A Bug/Poison type. Venonat can use Sleep Powder and Psychic moves. Mediocre stats overall."},
  {"name": "venomoth", "types": ["bug", "poison"], "hp": 70, "atk": 65, "def": 60, "spa": 90, "spd": 75, "spe": 90,
    "desc": "A fast Bug/Poison type with good Special. Venomoth has Sleep Powder, Psychic, and Stun Spore. Decent speed lets it move first with status moves."},
  {"name": "diglett", "types": ["ground"], "hp": 10, "atk": 55, "def": 25, "spa": 35, "spd": 45, "spe": 95,
    "desc": "An extremely fast but incredibly frail Ground type. Diglett has Earthquake and high Speed but the lowest HP in the game."},
  {"name": "dugtrio", "types": ["ground"], "hp": 35, "atk": 80, "def": 50, "spa": 50, "spd": 70, "spe": 120,
    "desc": "One of the fastest Pokémon in Gen 1 with base 120 Speed. Dugtrio has Earthquake and Slash but is very frail. Used as a revenge killer."},
  {"name": "meowth", "types": ["normal"], "hp": 40, "atk": 45, "def": 35, "spa": 40, "spd": 40, "spe": 90,
    "desc": "A fast Normal type. Meowth is frail but can use Slash, which has a very high critical hit rate in Gen 1."},
  {"name": "persian", "types": ["normal"], "hp": 65, "atk": 70, "def": 60, "spa": 65, "spd": 65, "spe": 115,
    "desc": "A very fast Normal type. Persian has Slash with a high crit rate in Gen 1 and can use Hyper Beam. Base 115 Speed is excellent."},
  {"name": "psyduck", "types": ["water"], "hp": 50, "atk": 52, "def": 48, "spa": 65, "spd": 50, "spe": 55,
    "desc": "A Water type with decent Special. Psyduck can use Surf and Psychic-type moves."},
  {"name": "golduck", "types": ["water"], "hp": 80, "atk": 82, "def": 78, "spa": 95, "spd": 80, "spe": 85,
    "desc": "A fast Water type with good Special. Golduck has Surf, Ice Beam, and Psychic. Decent but outclassed by Starmie and Slowbro."},
  {"name": "mankey", "types": ["fighting"], "hp": 40, "atk": 80, "def": 35, "spa": 35, "spd": 45, "spe": 70,
    "desc": "A Fighting type with high Attack. Mankey is frail but hits hard with Low Kick and Karate Chop."},
  {"name": "primeape", "types": ["fighting"], "hp": 65, "atk": 105, "def": 60, "spa": 60, "spd": 70, "spe": 95,
    "desc": "A fast Fighting type with high Attack. Primeape has Submission and Rock Slide. Good speed but relatively frail."},
  {"name": "growlithe", "types": ["fire"], "hp": 55, "atk": 70, "def": 45, "spa": 70, "spd": 50, "spe": 60,
    "desc": "A Fire type puppy. Growlithe has decent mixed attacking stats and evolves into the powerful Arcanine."},
  {"name": "arcanine", "types": ["fire"], "hp": 90, "atk": 110, "def": 80, "spa": 100, "spd": 80, "spe": 95,
    "desc": "A powerful Fire type with great all-around stats. Arcanine has Fire Blast, Body Slam, and high Attack and Special. One of the best Fire types in Gen 1."},
  {"name": "poliwag", "types": ["water"], "hp": 40, "atk": 50, "def": 40, "spa": 40, "spd": 40, "spe": 90,
    "desc": "A Water type tadpole. Poliwag is fast for its power level and can use Hypnosis and Water Gun."},
  {"name": "poliwhirl", "types": ["water"], "hp": 65, "atk": 65, "def": 65, "spa": 50, "spd": 50, "spe": 90,
    "desc": "The evolved form of Poliwag. Poliwhirl has Hypnosis, Surf, and Amnesia. Decent but outclassed."},
  {"name": "poliwrath", "types": ["water", "fighting"], "hp": 90, "atk": 85, "def": 95, "spa": 70, "spd": 90, "spe": 70,
    "desc": "A bulky Water/Fighting type. Poliwrath has Hypnosis, Surf, Submission, and great physical bulk. Amnesia makes it very tanky on the special side."},
  {"name": "abra", "types": ["psychic"], "hp": 25, "atk": 20, "def": 15, "spa": 105, "spd": 55, "spe": 90,
    "desc": "A Psychic type with high Special and Speed but almost no bulk. Abra only knows Teleport initially."},
  {"name": "kadabra", "types": ["psychic"], "hp": 40, "atk": 35, "def": 30, "spa": 120, "spd": 70, "spe": 105,
    "desc": "A fast Psychic type with very high Special. Kadabra has Psychic, Thunder Wave, and Recover. Glass cannon."},
  {"name": "alakazam", "types": ["psychic"], "hp": 55, "atk": 50, "def": 45, "spa": 135, "spd": 85, "spe": 120,
    "desc": "One of the best Pokémon in Gen 1. Alakazam has the highest Special and Speed among non-legendaries. Psychic, Thunder Wave, Recover, and Seismic Toss. Psychic types are dominant in Gen 1 due to no good counters."},
  {"name": "machop", "types": ["fighting"], "hp": 70, "atk": 80, "def": 50, "spa": 35, "spd": 35, "spe": 35,
    "desc": "A Fighting type with decent Attack. Machop is slow with poor Special but evolves into the strong Machamp."},
  {"name": "machoke", "types": ["fighting"], "hp": 80, "atk": 100, "def": 70, "spa": 50, "spd": 60, "spe": 45,
    "desc": "The evolved form of Machop. Machoke has high Attack and uses Submission and Earthquake."},
  {"name": "machamp", "types": ["fighting"], "hp": 90, "atk": 130, "def": 80, "spa": 65, "spd": 85, "spe": 55,
    "desc": "A powerful Fighting type with enormous Attack. Machamp uses Submission, Earthquake, and Hyper Beam. Slow but extremely strong physically."},
  {"name": "bellsprout", "types": ["grass", "poison"], "hp": 50, "atk": 75, "def": 35, "spa": 70, "spd": 30, "spe": 40,
    "desc": "A Grass/Poison type. Bellsprout has Sleep Powder and Razor Leaf. Frail but has decent Attack."},
  {"name": "weepinbell", "types": ["grass", "poison"], "hp": 65, "atk": 90, "def": 50, "spa": 85, "spd": 45, "spe": 55,
    "desc": "The evolved form of Bellsprout. Weepinbell has improved stats with Sleep Powder and Razor Leaf."},
  {"name": "victreebel", "types": ["grass", "poison"], "hp": 80, "atk": 105, "def": 65, "spa": 100, "spd": 60, "spe": 70,
    "desc": "A Grass/Poison type with strong Attack and Special. Victreebel uses Sleep Powder, Razor Leaf, and Swords Dance. A potent sweeper when given the chance."},
  {"name": "tentacool", "types": ["water", "poison"], "hp": 40, "atk": 40, "def": 35, "spa": 50, "spd": 100, "spe": 70,
    "desc": "A Water/Poison type with high Special Defense. Tentacool can use Surf and Sludge."},
  {"name": "tentacruel", "types": ["water", "poison"], "hp": 80, "atk": 70, "def": 65, "spa": 80, "spd": 120, "spe": 100,
    "desc": "A fast Water/Poison type with excellent Special. Tentacruel has Surf, Blizzard, and Wrap. High speed and special bulk make it a solid choice."},
  {"name": "geodude", "types": ["rock", "ground"], "hp": 40, "atk": 80, "def": 100, "spa": 30, "spd": 30, "spe": 20,
    "desc": "A Rock/Ground type with high physical Defense. Geodude has Earthquake and Rock Slide but is very slow with abysmal Special."},
  {"name": "graveler", "types": ["rock", "ground"], "hp": 55, "atk": 95, "def": 115, "spa": 45, "spd": 45, "spe": 35,
    "desc": "The evolved form of Geodude. Graveler has Earthquake, Rock Slide, and Explosion. Incredible physical bulk but terrible Special and Speed."},
  {"name": "golem", "types": ["rock", "ground"], "hp": 80, "atk": 110, "def": 130, "spa": 55, "spd": 65, "spe": 45,
    "desc": "A Rock/Ground type with massive Defense and Attack. Golem has Earthquake, Rock Slide, and Explosion. Crippled by terrible Special and low Speed."},
  {"name": "ponyta", "types": ["fire"], "hp": 50, "atk": 85, "def": 55, "spa": 65, "spd": 65, "spe": 90,
    "desc": "A fast Fire type. Ponyta has Fire Blast and Agility. Good Speed but limited movepool."},
  {"name": "rapidash", "types": ["fire"], "hp": 65, "atk": 100, "def": 70, "spa": 80, "spd": 80, "spe": 105,
    "desc": "A very fast Fire type. Rapidash has Fire Blast, Agility, and Hyper Beam. High Speed and decent Attack but limited coverage."},
  {"name": "slowpoke", "types": ["water", "psychic"], "hp": 90, "atk": 65, "def": 65, "spa": 40, "spd": 40, "spe": 15,
    "desc": "A bulky Water/Psychic type. Slowpoke is extremely slow but has good HP and evolves into the excellent Slowbro."},
  {"name": "slowbro", "types": ["water", "psychic"], "hp": 95, "atk": 75, "def": 110, "spa": 100, "spd": 80, "spe": 30,
    "desc": "One of the best Pokémon in Gen 1. Slowbro has excellent Defense, high Special, and access to Amnesia, Psychic, Surf, and Thunder Wave. Amnesia doubles Special making it nearly unbreakable."},
  {"name": "magnemite", "types": ["electric"], "hp": 25, "atk": 35, "def": 70, "spa": 95, "spd": 55, "spe": 45,
    "desc": "An Electric type with decent Special and Defense. Magnemite has Thunderbolt and Thunder Wave."},
  {"name": "magneton", "types": ["electric"], "hp": 50, "atk": 60, "def": 95, "spa": 120, "spd": 70, "spe": 70,
    "desc": "An Electric type with very high Special and good Defense. Magneton has Thunderbolt and Thunder Wave. Decent but lacks coverage moves."},
  {"name": "farfetchd", "types": ["normal", "flying"], "hp": 52, "atk": 65, "def": 55, "spa": 58, "spd": 62, "spe": 60,
    "desc": "A Normal/Flying type with mediocre stats all around. Farfetch'd has Slash and Swords Dance but is generally outclassed."},
  {"name": "doduo", "types": ["normal", "flying"], "hp": 35, "atk": 85, "def": 45, "spa": 35, "spd": 35, "spe": 75,
    "desc": "A Normal/Flying type with decent Attack and Speed. Doduo has Drill Peck and evolves into Dodrio."},
  {"name": "dodrio", "types": ["normal", "flying"], "hp": 60, "atk": 110, "def": 70, "spa": 60, "spd": 60, "spe": 100,
    "desc": "A fast and powerful Normal/Flying type. Dodrio has Drill Peck, Body Slam, and Hyper Beam with base 110 Attack and 100 Speed."},
  {"name": "seel", "types": ["water"], "hp": 65, "atk": 45, "def": 55, "spa": 45, "spd": 70, "spe": 45,
    "desc": "A Water type seal. Seel has Surf and Ice moves. Mediocre stats overall."},
  {"name": "dewgong", "types": ["water", "ice"], "hp": 90, "atk": 70, "def": 80, "spa": 70, "spd": 95, "spe": 70,
    "desc": "A bulky Water/Ice type. Dewgong has Surf, Ice Beam, and Rest. Good mixed bulk but low offensive pressure. Can wall certain threats."},
  {"name": "grimer", "types": ["poison"], "hp": 80, "atk": 80, "def": 50, "spa": 40, "spd": 50, "spe": 25,
    "desc": "A Poison type with decent HP and Attack but very slow. Grimer uses Sludge and Explosion."},
  {"name": "muk", "types": ["poison"], "hp": 105, "atk": 105, "def": 75, "spa": 65, "spd": 100, "spe": 50,
    "desc": "A bulky Poison type with high HP and Attack. Muk has Sludge, Explosion, and decent special bulk. Slow but can take hits."},
  {"name": "shellder", "types": ["water"], "hp": 30, "atk": 65, "def": 100, "spa": 45, "spd": 25, "spe": 40,
    "desc": "A Water type with extremely high Defense. Shellder has Clamp for trapping and evolves into Cloyster."},
  {"name": "cloyster", "types": ["water", "ice"], "hp": 50, "atk": 95, "def": 180, "spa": 85, "spd": 45, "spe": 70,
    "desc": "A Water/Ice type with the highest Defense in Gen 1 at base 180. Cloyster has Clamp, Blizzard, and Explosion. Paper-thin Special defense but physically almost impenetrable."},
  {"name": "gastly", "types": ["ghost", "poison"], "hp": 30, "atk": 35, "def": 30, "spa": 100, "spd": 35, "spe": 80,
    "desc": "A Ghost/Poison type with high Special. Gastly has Hypnosis, Night Shade, and Thunderbolt. Very frail but fast."},
  {"name": "haunter", "types": ["ghost", "poison"], "hp": 45, "atk": 50, "def": 45, "spa": 115, "spd": 55, "spe": 95,
    "desc": "A fast Ghost/Poison type with high Special. Haunter has Hypnosis, Thunderbolt, and Psychic. Immune to Normal and Fighting moves. In Gen 1, Ghost doesn't actually hit Psychic super-effectively due to a bug."},
  {"name": "gengar", "types": ["ghost", "poison"], "hp": 60, "atk": 65, "def": 60, "spa": 130, "spd": 75, "spe": 110,
    "desc": "A top-tier Pokémon in Gen 1. Gengar has excellent Special and Speed with Hypnosis, Thunderbolt, and Explosion. Immune to Normal and Fighting. Excellent as a fast sleeper and special attacker."},
  {"name": "onix", "types": ["rock", "ground"], "hp": 35, "atk": 45, "def": 160, "spa": 30, "spd": 45, "spe": 70,
    "desc": "A Rock/Ground type with extreme Defense but terrible stats everywhere else. Onix has the lowest Attack of any fully evolved Pokémon and terrible Special."},
  {"name": "drowzee", "types": ["psychic"], "hp": 60, "atk": 48, "def": 45, "spa": 43, "spd": 90, "spe": 42,
    "desc": "A Psychic type. Drowzee has Hypnosis and Psychic but mediocre stats overall."},
  {"name": "hypno", "types": ["psychic"], "hp": 85, "atk": 73, "def": 67, "spa": 73, "spd": 115, "spe": 67,
    "desc": "A bulky Psychic type with good Special Defense. Hypno has Hypnosis, Psychic, and Thunder Wave. Decent but outclassed by Alakazam and Starmie."},
  {"name": "krabby", "types": ["water"], "hp": 30, "atk": 105, "def": 90, "spa": 25, "spd": 25, "spe": 50,
    "desc": "A Water type with very high Attack for its stage. Krabby has Crabhammer and Swords Dance."},
  {"name": "kingler", "types": ["water"], "hp": 55, "atk": 130, "def": 115, "spa": 50, "spd": 50, "spe": 75,
    "desc": "A Water type with monstrous Attack and Defense. Kingler has Crabhammer and Swords Dance but terrible Special makes its Water STAB weak since Surf uses the Special stat in Gen 1."},
  {"name": "voltorb", "types": ["electric"], "hp": 40, "atk": 30, "def": 50, "spa": 55, "spd": 55, "spe": 100,
    "desc": "A fast Electric type. Voltorb has Thunderbolt, Thunder Wave, and Explosion."},
  {"name": "electrode", "types": ["electric"], "hp": 60, "atk": 50, "def": 70, "spa": 80, "spd": 80, "spe": 140,
    "desc": "The fastest Pokémon in Gen 1 with base 140 Speed. Electrode has Thunderbolt, Thunder Wave, and Explosion. Low power but incredible speed."},
  {"name": "exeggcute", "types": ["grass", "psychic"], "hp": 60, "atk": 40, "def": 80, "spa": 60, "spd": 45, "spe": 40,
    "desc": "A Grass/Psychic type. Exeggcute has Sleep Powder and Psychic. Decent Defense."},
  {"name": "exeggutor", "types": ["grass", "psychic"], "hp": 95, "atk": 95, "def": 85, "spa": 125, "spd": 65, "spe": 55,
    "desc": "A top-tier Pokémon in Gen 1. Exeggutor has enormous Special, Sleep Powder, Psychic, and Mega Drain/Solar Beam. Grass/Psychic is a great offensive typing. One of the best sleepers and special attackers."},
  {"name": "cubone", "types": ["ground"], "hp": 50, "atk": 50, "def": 95, "spa": 40, "spd": 50, "spe": 35,
    "desc": "A Ground type with decent Defense. Cubone has Earthquake and Bonemerang."},
  {"name": "marowak", "types": ["ground"], "hp": 60, "atk": 80, "def": 110, "spa": 50, "spd": 80, "spe": 45,
    "desc": "A Ground type with excellent Defense. Marowak has Earthquake and Bonemerang. Decent physically but slow with mediocre Special."},
  {"name": "hitmonlee", "types": ["fighting"], "hp": 50, "atk": 120, "def": 53, "spa": 35, "spd": 110, "spe": 87,
    "desc": "A Fighting type with very high Attack. Hitmonlee has Hi Jump Kick and decent Speed but is frail."},
  {"name": "hitmonchan", "types": ["fighting"], "hp": 50, "atk": 105, "def": 79, "spa": 35, "spd": 110, "spe": 76,
    "desc": "A Fighting type with high Attack and decent Defense. Hitmonchan has the elemental punches (Thunder, Ice, Fire Punch) for coverage."},
  {"name": "lickitung", "types": ["normal"], "hp": 90, "atk": 55, "def": 75, "spa": 60, "spd": 75, "spe": 30,
    "desc": "A Normal type with good HP and decent bulk. Lickitung has Swords Dance but is very slow and lacks power."},
  {"name": "koffing", "types": ["poison"], "hp": 40, "atk": 65, "def": 95, "spa": 60, "spd": 45, "spe": 35,
    "desc": "A Poison type with high Defense. Koffing has Sludge, Explosion, and Thunderbolt."},
  {"name": "weezing", "types": ["poison"], "hp": 65, "atk": 90, "def": 120, "spa": 85, "spd": 70, "spe": 60,
    "desc": "A bulky Poison type with excellent Defense. Weezing has Sludge, Explosion, and Thunderbolt. Good physical wall."},
  {"name": "rhyhorn", "types": ["ground", "rock"], "hp": 80, "atk": 85, "def": 95, "spa": 30, "spd": 30, "spe": 25,
    "desc": "A Ground/Rock type with decent Attack and Defense. Rhyhorn has Earthquake but is extremely slow with terrible Special."},
  {"name": "rhydon", "types": ["ground", "rock"], "hp": 105, "atk": 130, "def": 120, "spa": 45, "spd": 45, "spe": 40,
    "desc": "A powerful Ground/Rock type with massive Attack and Defense. Rhydon has Earthquake, Rock Slide, and Substitute. Crippled by abysmal Special—any Water or Ice special move will KO it."},
  {"name": "chansey", "types": ["normal"], "hp": 250, "atk": 5, "def": 5, "spa": 35, "spd": 105, "spe": 50,
    "desc": "The premier special wall in Gen 1 with an absurd 250 base HP. Chansey has Soft-Boiled for recovery, Thunder Wave, Seismic Toss, and Ice Beam. Dominates the Gen 1 metagame as a near-mandatory team member."},
  {"name": "tangela", "types": ["grass"], "hp": 65, "atk": 55, "def": 115, "spa": 100, "spd": 40, "spe": 60,
    "desc": "A Grass type with excellent Defense and good Special. Tangela has Sleep Powder, Mega Drain, and Stun Spore. A solid physical wall."},
  {"name": "kangaskhan", "types": ["normal"], "hp": 105, "atk": 95, "def": 80, "spa": 40, "spd": 80, "spe": 90,
    "desc": "A bulky Normal type with great HP, Attack, and Speed. Kangaskhan has Body Slam, Earthquake, and Hyper Beam. A solid all-around attacker."},
  {"name": "horsea", "types": ["water"], "hp": 30, "atk": 40, "def": 70, "spa": 70, "spd": 25, "spe": 60,
    "desc": "A Water type. Horsea has Surf and Agility. Low stats overall."},
  {"name": "seadra", "types": ["water"], "hp": 55, "atk": 65, "def": 95, "spa": 95, "spd": 45, "spe": 85,
    "desc": "A Water type with good Special and Defense. Seadra has Surf, Ice Beam, and Agility. Decent but lacks recovery."},
  {"name": "goldeen", "types": ["water"], "hp": 45, "atk": 67, "def": 60, "spa": 35, "spd": 50, "spe": 63,
    "desc": "A Water type. Goldeen has Waterfall and Agility but mediocre stats."},
  {"name": "seaking", "types": ["water"], "hp": 80, "atk": 92, "def": 65, "spa": 65, "spd": 80, "spe": 68,
    "desc": "A Water type with decent Attack. Seaking has Waterfall, Agility, and Hyper Beam. Outclassed by most other Water types."},
  {"name": "staryu", "types": ["water"], "hp": 30, "atk": 45, "def": 55, "spa": 70, "spd": 55, "spe": 85,
    "desc": "A Water type with decent Special and Speed. Staryu can use Surf and Recover. Evolves into the excellent Starmie."},
  {"name": "starmie", "types": ["water", "psychic"], "hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115,
    "desc": "One of the best Pokémon in Gen 1. Starmie has excellent Speed and Special with Surf, Psychic, Thunderbolt, Ice Beam, Recover, and Thunder Wave. Water/Psychic typing is incredible offensively and defensively. A metagame staple."},
  {"name": "mrmime", "types": ["psychic"], "hp": 40, "atk": 45, "def": 65, "spa": 100, "spd": 120, "spe": 90,
    "desc": "A Psychic type with high Special. Mr. Mime has Psychic, Thunder Wave, and decent Speed. Outclassed by Alakazam but still usable."},
  {"name": "scyther", "types": ["bug", "flying"], "hp": 70, "atk": 110, "def": 80, "spa": 55, "spd": 80, "spe": 105,
    "desc": "A fast Bug/Flying type with high Attack. Scyther has Slash with its high crit rate and Swords Dance. Lacks STAB moves in Gen 1 since there are no good Bug-type attacks."},
  {"name": "jynx", "types": ["ice", "psychic"], "hp": 65, "atk": 50, "def": 35, "spa": 115, "spd": 95, "spe": 95,
    "desc": "An Ice/Psychic type with excellent Special and Speed. Jynx has Blizzard, Psychic, and Lovely Kiss for sleep. A potent offensive threat and sleeper."},
  {"name": "electabuzz", "types": ["electric"], "hp": 65, "atk": 83, "def": 57, "spa": 95, "spd": 85, "spe": 105,
    "desc": "A fast Electric type with good Special. Electabuzz has Thunderbolt, Psychic, and Thunder Wave. Decent but outclassed by Zapdos and Jolteon."},
  {"name": "magmar", "types": ["fire"], "hp": 65, "atk": 95, "def": 57, "spa": 100, "spd": 85, "spe": 93,
    "desc": "A Fire type with decent Special and Attack. Magmar has Fire Blast, Psychic, and Confuse Ray. Decent but outclassed by other Fire types."},
  {"name": "pinsir", "types": ["bug"], "hp": 65, "atk": 125, "def": 100, "spa": 55, "spd": 70, "spe": 85,
    "desc": "A Bug type with massive Attack and great Defense. Pinsir has Swords Dance and Submission but lacks good STAB in Gen 1 due to no strong Bug moves."},
  {"name": "tauros", "types": ["normal"], "hp": 75, "atk": 100, "def": 95, "spa": 40, "spd": 70, "spe": 110,
    "desc": "The best Pokémon in Gen 1. Tauros has outstanding Attack, Defense, and Speed with Body Slam, Hyper Beam, Earthquake, and Blizzard. Dominates the metagame with almost no drawbacks."},
  {"name": "magikarp", "types": ["water"], "hp": 20, "atk": 10, "def": 55, "spa": 15, "spd": 20, "spe": 80,
    "desc": "Famously the weakest Pokémon. Magikarp only learns Splash and Tackle. It evolves into the mighty Gyarados."},
  {"name": "gyarados", "types": ["water", "flying"], "hp": 95, "atk": 125, "def": 79, "spa": 60, "spd": 100, "spe": 81,
    "desc": "A Water/Flying type with enormous Attack. Gyarados has Hydro Pump, Thunderbolt, and Hyper Beam. However, its Water STAB is special-based and its Special stat is only 100. Still a solid threat."},
  {"name": "lapras", "types": ["water", "ice"], "hp": 130, "atk": 85, "def": 80, "spa": 85, "spd": 95, "spe": 60,
    "desc": "A bulky Water/Ice type with great HP. Lapras has Surf, Ice Beam, Thunderbolt, and Body Slam. Excellent mixed bulk and coverage. A staple on many Gen 1 teams."},
  {"name": "ditto", "types": ["normal"], "hp": 48, "atk": 48, "def": 48, "spa": 48, "spd": 48, "spe": 48,
    "desc": "A Normal type that uses Transform to copy the opponent exactly, including moves, stats, and typing. Ditto is unique but generally unreliable competitively."},
  {"name": "eevee", "types": ["normal"], "hp": 55, "atk": 55, "def": 50, "spa": 45, "spd": 65, "spe": 55,
    "desc": "A Normal type. Eevee evolves into Vaporeon, Jolteon, or Flareon in Gen 1."},
  {"name": "vaporeon", "types": ["water"], "hp": 130, "atk": 65, "def": 60, "spa": 110, "spd": 95, "spe": 65,
    "desc": "A bulky Water type with huge HP and high Special. Vaporeon has Surf, Ice Beam, and Acid Armor. Excellent special tank."},
  {"name": "jolteon", "types": ["electric"], "hp": 65, "atk": 65, "def": 60, "spa": 110, "spd": 95, "spe": 130,
    "desc": "One of the fastest Pokémon in Gen 1. Jolteon has excellent Special and Speed with Thunderbolt, Thunder Wave, and Pin Missile. A premier Electric type."},
  {"name": "flareon", "types": ["fire"], "hp": 65, "atk": 130, "def": 60, "spa": 95, "spd": 110, "spe": 65,
    "desc": "A Fire type with massive Attack but poor Speed. Flareon suffers in Gen 1 because Fire Blast is special and there are no good physical Fire moves. Largely outclassed."},
  {"name": "porygon", "types": ["normal"], "hp": 65, "atk": 60, "def": 70, "spa": 85, "spd": 75, "spe": 40,
    "desc": "A Normal type with decent Special. Porygon has Thunderbolt, Ice Beam, and Recover. Slow but has good coverage."},
  {"name": "omanyte", "types": ["rock", "water"], "hp": 35, "atk": 40, "def": 100, "spa": 90, "spd": 55, "spe": 35,
    "desc": "A Rock/Water fossil type. Omanyte has high Defense and decent Special. Evolves into Omastar."},
  {"name": "omastar", "types": ["rock", "water"], "hp": 70, "atk": 60, "def": 125, "spa": 115, "spd": 70, "spe": 55,
    "desc": "A Rock/Water type with excellent Defense and Special. Omastar has Surf, Ice Beam, and Seismic Toss. Great bulk but slow with a crippling Grass weakness."},
  {"name": "kabuto", "types": ["rock", "water"], "hp": 30, "atk": 80, "def": 90, "spa": 55, "spd": 45, "spe": 55,
    "desc": "A Rock/Water fossil type. Kabuto has decent Attack and Defense. Evolves into Kabutops."},
  {"name": "kabutops", "types": ["rock", "water"], "hp": 60, "atk": 115, "def": 105, "spa": 65, "spd": 70, "spe": 80,
    "desc": "A Rock/Water type with high Attack and Defense. Kabutops has Swords Dance and Slash. More offensive than Omastar but still slow."},
  {"name": "aerodactyl", "types": ["rock", "flying"], "hp": 80, "atk": 105, "def": 65, "spa": 60, "spd": 75, "spe": 130,
    "desc": "A very fast Rock/Flying type. Aerodactyl has base 130 Speed with Hyper Beam and Sky Attack. Frail on the special side but one of the fastest Pokémon available."},
  {"name": "snorlax", "types": ["normal"], "hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30,
    "desc": "One of the best Pokémon in Gen 1. Snorlax has enormous HP and Attack with Body Slam, Earthquake, Hyper Beam, Selfdestruct, and Amnesia. A dominant force that can both wall and sweep. Rest provides recovery."},
  {"name": "articuno", "types": ["ice", "flying"], "hp": 90, "atk": 85, "def": 100, "spa": 95, "spd": 125, "spe": 85,
    "desc": "A legendary Ice/Flying type. Articuno has Blizzard, Ice Beam, and Agility with excellent bulk. Decent but its typing gives it many weaknesses."},
  {"name": "zapdos", "types": ["electric", "flying"], "hp": 90, "atk": 90, "def": 85, "spa": 125, "spd": 90, "spe": 100,
    "desc": "One of the best Pokémon in Gen 1. Zapdos has excellent Special and Speed with Thunderbolt, Drill Peck, Thunder Wave, and Agility. Electric/Flying is great typing. A metagame staple."},
  {"name": "moltres", "types": ["fire", "flying"], "hp": 90, "atk": 100, "def": 90, "spa": 125, "spd": 85, "spe": 90,
    "desc": "A legendary Fire/Flying type with high Special. Moltres has Fire Blast and Agility. Decent but less useful than Zapdos due to Stealth Rock weakness and less useful typing."},
  {"name": "dratini", "types": ["dragon"], "hp": 41, "atk": 64, "def": 45, "spa": 50, "spd": 50, "spe": 50,
    "desc": "A Dragon type. Dratini is weak but evolves into Dragonair and eventually the powerful Dragonite."},
  {"name": "dragonair", "types": ["dragon"], "hp": 61, "atk": 84, "def": 65, "spa": 70, "spd": 70, "spe": 70,
    "desc": "The evolved form of Dratini. Dragonair has decent all-around stats and Thunder Wave. Evolves into Dragonite."},
  {"name": "dragonite", "types": ["dragon", "flying"], "hp": 91, "atk": 134, "def": 95, "spa": 100, "spd": 100, "spe": 80,
    "desc": "A powerful Dragon/Flying type with the highest Attack in Gen 1. Dragonite has Wrap, Hyper Beam, Blizzard, Thunderbolt, and Agility. Versatile but somewhat slow for a sweeper."},
  {"name": "mewtwo", "types": ["psychic"], "hp": 106, "atk": 110, "def": 90, "spa": 154, "spd": 90, "spe": 130,
    "desc": "The most powerful Pokémon in Gen 1, usually banned to Ubers. Mewtwo has monstrous Special and Speed with Psychic, Ice Beam, Thunderbolt, Amnesia, and Recover. Virtually no counters in standard play."},
  {"name": "mew", "types": ["psychic"], "hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100,
    "desc": "A Psychic type with perfectly balanced base 100 stats across the board. Mew can learn every TM and HM, giving it unmatched versatility. Usually banned to Ubers alongside Mewtwo."},
]

# ---------------------------------------------------------------------------
# Gen 1 Moves — all moves learnable in Gen 1 random battles
# Each entry includes: name, type, category, power, accuracy, PP, and a
# description of its effect.
# NOTE: In Gen 1, the "Special" stat governs both SpA and SpD, and there is
# no physical/special split — category is determined by the move's type.
# Physical types: Normal, Fighting, Poison, Ground, Flying, Bug, Rock, Ghost
# Special types: Water, Grass, Fire, Ice, Electric, Psychic, Dragon
# ---------------------------------------------------------------------------

GEN1_MOVES: List[Dict] = [
  # --- Normal-type moves ---
  {"name": "tackle", "type": "normal", "category": "physical", "power": 35, "accuracy": 95, "pp": 35,
    "desc": "A basic physical Normal-type attack. Low power, available to many Pokémon. No additional effects."},
  {"name": "pound", "type": "normal", "category": "physical", "power": 40, "accuracy": 100, "pp": 35,
    "desc": "A basic physical Normal-type attack. Slightly stronger than Tackle with better accuracy."},
  {"name": "scratch", "type": "normal", "category": "physical", "power": 40, "accuracy": 100, "pp": 35,
    "desc": "A basic physical Normal-type attack. No additional effects."},
  {"name": "bodyslam", "type": "normal", "category": "physical", "power": 85, "accuracy": 100, "pp": 15,
    "desc": "A strong physical Normal-type attack with a 30% chance to paralyze the target. One of the best moves in Gen 1 due to reliable damage and paralysis chance."},
  {"name": "slash", "type": "normal", "category": "physical", "power": 70, "accuracy": 100, "pp": 20,
    "desc": "A physical Normal-type attack with a very high critical hit ratio. In Gen 1, crits are based on Speed, making this devastating on fast Pokémon like Persian."},
  {"name": "strength", "type": "normal", "category": "physical", "power": 80, "accuracy": 100, "pp": 15,
    "desc": "A reliable physical Normal-type attack. No additional effects. Used as a solid damage option."},
  {"name": "headbutt", "type": "normal", "category": "physical", "power": 70, "accuracy": 100, "pp": 15,
    "desc": "A physical Normal-type attack with a 30% chance to cause the target to flinch."},
  {"name": "hyperbeam", "type": "normal", "category": "physical", "power": 150, "accuracy": 90, "pp": 5,
    "desc": "The most powerful Normal-type attack. Requires a recharge turn after use. In Gen 1, if Hyper Beam KOs the target, no recharge is needed, making it the best finisher."},
  {"name": "doubleedge", "type": "normal", "category": "physical", "power": 100, "accuracy": 100, "pp": 15,
    "desc": "A powerful physical Normal-type attack that deals 25% of the damage dealt as recoil to the user."},
  {"name": "takedown", "type": "normal", "category": "physical", "power": 90, "accuracy": 85, "pp": 20,
    "desc": "A physical Normal-type attack that deals 25% recoil damage. Lower accuracy than Double-Edge."},
  {"name": "quickattack", "type": "normal", "category": "physical", "power": 40, "accuracy": 100, "pp": 30,
    "desc": "A weak physical Normal-type attack that always goes first (priority +1). Useful for picking off weakened foes."},
  {"name": "rage", "type": "normal", "category": "physical", "power": 20, "accuracy": 100, "pp": 20,
    "desc": "A weak Normal-type attack. The user's Attack increases each time it is hit. In Gen 1, once Rage is selected, the user is locked into it."},
  {"name": "wrap", "type": "normal", "category": "physical", "power": 15, "accuracy": 85, "pp": 20,
    "desc": "A trapping Normal-type move that prevents the target from moving for 2-5 turns while dealing damage each turn. Extremely powerful in Gen 1 because the trapped Pokémon cannot act at all."},
  {"name": "bind", "type": "normal", "category": "physical", "power": 15, "accuracy": 75, "pp": 20,
    "desc": "A trapping Normal-type move similar to Wrap. Prevents the target from moving for 2-5 turns. Lower accuracy than Wrap."},
  {"name": "clamp", "type": "water", "category": "special", "power": 35, "accuracy": 75, "pp": 10,
    "desc": "A trapping Water-type move. Prevents the target from moving for 2-5 turns while dealing damage. Used by Cloyster."},
  {"name": "slam", "type": "normal", "category": "physical", "power": 80, "accuracy": 75, "pp": 20,
    "desc": "A physical Normal-type attack with mediocre accuracy. Body Slam is usually preferred."},
  {"name": "stomp", "type": "normal", "category": "physical", "power": 65, "accuracy": 100, "pp": 20,
    "desc": "A physical Normal-type attack with a 30% chance to flinch."},
  {"name": "megakick", "type": "normal", "category": "physical", "power": 120, "accuracy": 75, "pp": 5,
    "desc": "A very powerful physical Normal-type attack with shaky accuracy."},
  {"name": "megapunch", "type": "normal", "category": "physical", "power": 80, "accuracy": 85, "pp": 20,
    "desc": "A physical Normal-type attack. Outclassed by Body Slam due to lower power and no paralysis chance."},
  {"name": "cometpunch", "type": "normal", "category": "physical", "power": 18, "accuracy": 85, "pp": 15,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times."},
  {"name": "furyattack", "type": "normal", "category": "physical", "power": 15, "accuracy": 85, "pp": 20,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times."},
  {"name": "furyswipes", "type": "normal", "category": "physical", "power": 18, "accuracy": 80, "pp": 15,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times."},
  {"name": "doubleslap", "type": "normal", "category": "physical", "power": 15, "accuracy": 85, "pp": 10,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times."},
  {"name": "barrage", "type": "normal", "category": "physical", "power": 15, "accuracy": 85, "pp": 20,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times. Primarily used by Exeggcute and Exeggutor."},
  {"name": "spikecannon", "type": "normal", "category": "physical", "power": 20, "accuracy": 100, "pp": 15,
    "desc": "A multi-hit Normal-type attack that hits 2-5 times."},
  {"name": "pinmissile", "type": "bug", "category": "physical", "power": 14, "accuracy": 85, "pp": 20,
    "desc": "A multi-hit Bug-type attack that hits 2-5 times. Used by Jolteon for coverage."},
  {"name": "selfdestruct", "type": "normal", "category": "physical", "power": 200, "accuracy": 100, "pp": 5,
    "desc": "An extremely powerful Normal-type attack that causes the user to faint. In Gen 1, the target's Defense is halved during damage calculation, effectively making it even stronger."},
  {"name": "explosion", "type": "normal", "category": "physical", "power": 250, "accuracy": 100, "pp": 5,
    "desc": "The most powerful move in the game. Causes the user to faint. In Gen 1, the target's Defense is halved during damage calculation. Used as a last resort to remove a threatening opponent."},
  {"name": "superfang", "type": "normal", "category": "physical", "power": 0, "accuracy": 90, "pp": 10,
    "desc": "Always deals damage equal to half the target's current HP. Ignores type resistances. Very useful against bulky Pokémon."},
  {"name": "hornattack", "type": "normal", "category": "physical", "power": 65, "accuracy": 100, "pp": 25,
    "desc": "A basic physical Normal-type attack."},
  {"name": "horndrill", "type": "normal", "category": "physical", "power": 0, "accuracy": 30, "pp": 5,
    "desc": "A one-hit KO move. Always fails against faster Pokémon in Gen 1. 30% accuracy otherwise."},
  {"name": "guillotine", "type": "normal", "category": "physical", "power": 0, "accuracy": 30, "pp": 5,
    "desc": "A one-hit KO move. Always fails against faster Pokémon in Gen 1. 30% accuracy."},
  {"name": "fissure", "type": "ground", "category": "physical", "power": 0, "accuracy": 30, "pp": 5,
    "desc": "A one-hit KO Ground-type move. Always fails against faster Pokémon in Gen 1."},
  {"name": "skullbash", "type": "normal", "category": "physical", "power": 100, "accuracy": 100, "pp": 15,
    "desc": "A two-turn Normal-type attack. The user charges on the first turn and attacks on the second. Generally not worth the two-turn setup."},
  {"name": "dizzypunch", "type": "normal", "category": "physical", "power": 70, "accuracy": 100, "pp": 10,
    "desc": "A physical Normal-type attack. No additional effects in Gen 1."},
  {"name": "payday", "type": "normal", "category": "physical", "power": 40, "accuracy": 100, "pp": 20,
    "desc": "A weak physical Normal-type attack. No competitive use."},
  {"name": "triattack", "type": "normal", "category": "physical", "power": 80, "accuracy": 100, "pp": 10,
    "desc": "A physical Normal-type attack. No additional effects in Gen 1. Used by Dodrio and Porygon."},
  # --- Fighting-type moves ---
  {"name": "karatechop", "type": "fighting", "category": "physical", "power": 50, "accuracy": 100, "pp": 25,
    "desc": "A physical Fighting-type attack with a high critical hit ratio."},
  {"name": "submission", "type": "fighting", "category": "physical", "power": 80, "accuracy": 80, "pp": 25,
    "desc": "A physical Fighting-type attack that deals 25% recoil to the user. The main Fighting STAB option despite mediocre accuracy."},
  {"name": "lowkick", "type": "fighting", "category": "physical", "power": 50, "accuracy": 90, "pp": 20,
    "desc": "A physical Fighting-type attack. In Gen 1, it has a 30% flinch chance. Fixed 50 base power."},
  {"name": "hijumpkick", "type": "fighting", "category": "physical", "power": 85, "accuracy": 90, "pp": 20,
    "desc": "A powerful physical Fighting-type attack. If it misses, the user takes crash damage equal to 1 HP in Gen 1."},
  {"name": "jumpkick", "type": "fighting", "category": "physical", "power": 70, "accuracy": 95, "pp": 25,
    "desc": "A physical Fighting-type attack. If it misses, the user takes crash damage."},
  {"name": "seismictoss", "type": "fighting", "category": "physical", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Always deals exactly damage equal to the user's level. At level 100, always does 100 damage. Ignores type. Used by Chansey and other Pokémon that lack offensive power."},
  {"name": "counter", "type": "fighting", "category": "physical", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Returns double the damage received from physical or Normal-type attacks. Priority -1. In Gen 1, Counter works against Normal and Fighting type moves."},
  {"name": "rollingkick", "type": "fighting", "category": "physical", "power": 60, "accuracy": 85, "pp": 15,
    "desc": "A physical Fighting-type attack with a 30% flinch chance."},
  {"name": "doublekick", "type": "fighting", "category": "physical", "power": 30, "accuracy": 100, "pp": 30,
    "desc": "A physical Fighting-type attack that hits twice. Each hit does 30 base power."},
  # --- Poison-type moves ---
  {"name": "poisonsting", "type": "poison", "category": "physical", "power": 15, "accuracy": 100, "pp": 35,
    "desc": "A weak Poison-type attack with a 30% chance to poison the target."},
  {"name": "sludge", "type": "poison", "category": "physical", "power": 65, "accuracy": 100, "pp": 20,
    "desc": "A physical Poison-type attack with a 30% chance to poison. The best Poison STAB in Gen 1."},
  {"name": "acid", "type": "poison", "category": "physical", "power": 40, "accuracy": 100, "pp": 30,
    "desc": "A Poison-type attack with a 10% chance to lower the target's Defense."},
  {"name": "toxic", "type": "poison", "category": "status", "power": 0, "accuracy": 85, "pp": 10,
    "desc": "Badly poisons the target, causing increasing damage each turn. In Gen 1, Toxic and Leech Seed interact to increase both their damage counters together, a powerful combo."},
  {"name": "poisonpowder", "type": "poison", "category": "status", "power": 0, "accuracy": 75, "pp": 35,
    "desc": "Poisons the target. Does not work on Poison types."},
  {"name": "smog", "type": "poison", "category": "physical", "power": 20, "accuracy": 70, "pp": 20,
    "desc": "A weak Poison-type attack with a 40% chance to poison. Poor accuracy and power."},
  {"name": "poisongas", "type": "poison", "category": "status", "power": 0, "accuracy": 55, "pp": 40,
    "desc": "Poisons the target. Very low accuracy makes it unreliable."},
  # --- Ground-type moves ---
  {"name": "earthquake", "type": "ground", "category": "physical", "power": 100, "accuracy": 100, "pp": 10,
    "desc": "The best Ground-type attack. High power, perfect accuracy. One of the most important coverage moves in Gen 1. Hits all grounded Pokémon."},
  {"name": "dig", "type": "ground", "category": "physical", "power": 100, "accuracy": 100, "pp": 10,
    "desc": "A two-turn Ground-type attack. The user digs underground on the first turn and attacks on the second. Predictable and generally inferior to Earthquake."},
  {"name": "boneclub", "type": "ground", "category": "physical", "power": 65, "accuracy": 85, "pp": 20,
    "desc": "A Ground-type attack with a 10% flinch chance. Used by Cubone and Marowak."},
  {"name": "bonemerang", "type": "ground", "category": "physical", "power": 50, "accuracy": 90, "pp": 10,
    "desc": "A Ground-type attack that hits twice. Total base power is 100, similar to Earthquake but split into two hits."},
  {"name": "sandattack", "type": "ground", "category": "status", "power": 0, "accuracy": 100, "pp": 15,
    "desc": "Lowers the target's accuracy by one stage. A useful disruption move."},
  # --- Flying-type moves ---
  {"name": "fly", "type": "flying", "category": "physical", "power": 70, "accuracy": 95, "pp": 15,
    "desc": "A two-turn Flying-type attack. User flies up on the first turn and attacks on the second."},
  {"name": "drillpeck", "type": "flying", "category": "physical", "power": 80, "accuracy": 100, "pp": 20,
    "desc": "A solid physical Flying-type attack. Good power and accuracy. The best Flying STAB in Gen 1."},
  {"name": "wingattack", "type": "flying", "category": "physical", "power": 35, "accuracy": 100, "pp": 35,
    "desc": "A weak physical Flying-type attack. Only 35 base power in Gen 1, much weaker than Drill Peck."},
  {"name": "skyattack", "type": "flying", "category": "physical", "power": 140, "accuracy": 90, "pp": 5,
    "desc": "A two-turn Flying-type attack. Very high power but the charging turn makes it predictable."},
  {"name": "mirrormove", "type": "flying", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Copies the last move used by the opponent. Can be useful but unreliable."},
  {"name": "peck", "type": "flying", "category": "physical", "power": 35, "accuracy": 100, "pp": 35,
    "desc": "A basic physical Flying-type attack. Weak."},
  {"name": "gust", "type": "normal", "category": "physical", "power": 40, "accuracy": 100, "pp": 35,
    "desc": "In Gen 1, Gust is a Normal-type move, not Flying. A basic attack with no additional effects."},
  # --- Bug-type moves ---
  {"name": "twineedle", "type": "bug", "category": "physical", "power": 25, "accuracy": 100, "pp": 20,
    "desc": "A Bug-type attack that hits twice. Each hit has a 20% chance to poison. Used by Beedrill."},
  {"name": "leechlife", "type": "bug", "category": "physical", "power": 20, "accuracy": 100, "pp": 15,
    "desc": "A weak Bug-type attack that heals the user for half the damage dealt. Very low power in Gen 1."},
  {"name": "stringshot", "type": "bug", "category": "status", "power": 0, "accuracy": 95, "pp": 40,
    "desc": "Lowers the target's Speed by one stage. Useful in Gen 1 since Speed affects critical hit rate."},
  # --- Rock-type moves ---
  {"name": "rockslide", "type": "rock", "category": "physical", "power": 75, "accuracy": 90, "pp": 10,
    "desc": "A physical Rock-type attack. The only usable Rock-type attack in Gen 1 besides Rock Throw."},
  {"name": "rockthrow", "type": "rock", "category": "physical", "power": 50, "accuracy": 65, "pp": 15,
    "desc": "A physical Rock-type attack with poor accuracy. Outclassed by Rock Slide."},
  # --- Ghost-type moves ---
  {"name": "nightshade", "type": "ghost", "category": "physical", "power": 0, "accuracy": 100, "pp": 15,
    "desc": "Always deals exactly damage equal to the user's level. Similar to Seismic Toss but Ghost-type. Does not affect Normal types."},
  {"name": "lick", "type": "ghost", "category": "physical", "power": 20, "accuracy": 100, "pp": 30,
    "desc": "A weak Ghost-type attack with a 30% paralysis chance. In Gen 1, Ghost does not hit Psychic super-effectively due to a bug."},
  {"name": "confuseray", "type": "ghost", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Confuses the target. Confused Pokémon have a 50% chance of hitting themselves each turn in Gen 1."},
  # --- Water-type moves ---
  {"name": "surf", "type": "water", "category": "special", "power": 95, "accuracy": 100, "pp": 15,
    "desc": "A powerful special Water-type attack. Reliable damage with perfect accuracy. The best general Water STAB."},
  {"name": "hydropump", "type": "water", "category": "special", "power": 120, "accuracy": 80, "pp": 5,
    "desc": "A very powerful special Water-type attack with shaky 80% accuracy. Higher power than Surf but less reliable."},
  {"name": "watergun", "type": "water", "category": "special", "power": 40, "accuracy": 100, "pp": 25,
    "desc": "A basic special Water-type attack. Low power."},
  {"name": "bubblebeam", "type": "water", "category": "special", "power": 65, "accuracy": 100, "pp": 20,
    "desc": "A special Water-type attack with a 10% chance to lower the target's Speed."},
  {"name": "bubble", "type": "water", "category": "special", "power": 20, "accuracy": 100, "pp": 30,
    "desc": "A very weak special Water-type attack."},
  {"name": "waterfall", "type": "water", "category": "special", "power": 80, "accuracy": 100, "pp": 15,
    "desc": "A special Water-type attack. In Gen 1, Waterfall is treated as special. Decent power."},
  {"name": "crabhammer", "type": "water", "category": "special", "power": 90, "accuracy": 85, "pp": 10,
    "desc": "A special Water-type attack with a high critical hit ratio. Used by Kingler. Despite Kingler's high Attack, Crabhammer uses the Special stat in Gen 1."},
  {"name": "withdraw", "type": "water", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Raises the user's Defense by one stage. A defensive setup move."},
  # --- Grass-type moves ---
  {"name": "razorleaf", "type": "grass", "category": "special", "power": 55, "accuracy": 95, "pp": 25,
    "desc": "A special Grass-type attack with a high critical hit ratio. The main Grass-type STAB move."},
  {"name": "solarbeam", "type": "grass", "category": "special", "power": 120, "accuracy": 100, "pp": 10,
    "desc": "A powerful Grass-type attack that requires a charging turn. Can be used instantly in sunny weather (not available in Gen 1 mechanics)."},
  {"name": "megadrain", "type": "grass", "category": "special", "power": 40, "accuracy": 100, "pp": 10,
    "desc": "A special Grass-type attack that heals the user for half the damage dealt."},
  {"name": "absorb", "type": "grass", "category": "special", "power": 20, "accuracy": 100, "pp": 20,
    "desc": "A weak special Grass-type attack that heals the user for half the damage dealt."},
  {"name": "vinewhip", "type": "grass", "category": "special", "power": 35, "accuracy": 100, "pp": 10,
    "desc": "A basic special Grass-type attack."},
  {"name": "petaldance", "type": "grass", "category": "special", "power": 70, "accuracy": 100, "pp": 20,
    "desc": "A special Grass-type attack that hits for 2-3 turns. Confuses the user afterward."},
  {"name": "leechseed", "type": "grass", "category": "status", "power": 0, "accuracy": 90, "pp": 10,
    "desc": "Plants a seed on the target that drains HP each turn, healing the user. Does not work on Grass types. In Gen 1, Leech Seed and Toxic counters interact, increasing both damage values."},
  {"name": "sleeppowder", "type": "grass", "category": "status", "power": 0, "accuracy": 75, "pp": 15,
    "desc": "Puts the target to sleep. Sleep is the most powerful status in Gen 1 because a Pokémon must wake up AND attack on the same turn. 75% accuracy."},
  {"name": "stunspore", "type": "grass", "category": "status", "power": 0, "accuracy": 75, "pp": 30,
    "desc": "Paralyzes the target. Paralysis halves Speed and has a 25% full-paralysis chance. Very strong in Gen 1 because Speed affects crit rate."},
  {"name": "spore", "type": "grass", "category": "status", "power": 0, "accuracy": 100, "pp": 15,
    "desc": "Puts the target to sleep with 100% accuracy. The best sleep move in the game. Only learned by Parasect in Gen 1."},
  # --- Fire-type moves ---
  {"name": "fireblast", "type": "fire", "category": "special", "power": 120, "accuracy": 85, "pp": 5,
    "desc": "A very powerful special Fire-type attack with a 30% burn chance. The best Fire STAB move in Gen 1."},
  {"name": "flamethrower", "type": "fire", "category": "special", "power": 95, "accuracy": 100, "pp": 15,
    "desc": "A reliable special Fire-type attack with a 10% burn chance. More accurate than Fire Blast."},
  {"name": "ember", "type": "fire", "category": "special", "power": 40, "accuracy": 100, "pp": 25,
    "desc": "A basic special Fire-type attack with a 10% burn chance."},
  {"name": "firespin", "type": "fire", "category": "special", "power": 15, "accuracy": 70, "pp": 15,
    "desc": "A trapping Fire-type move. Like Wrap, prevents the target from moving for 2-5 turns. Low accuracy."},
  {"name": "firepunch", "type": "fire", "category": "special", "power": 75, "accuracy": 100, "pp": 15,
    "desc": "A special Fire-type attack with a 10% burn chance."},
  # --- Ice-type moves ---
  {"name": "blizzard", "type": "ice", "category": "special", "power": 120, "accuracy": 90, "pp": 5,
    "desc": "A very powerful special Ice-type attack with a 10% freeze chance. In Gen 1, Blizzard has 90% accuracy, making it the preferred Ice attack over Ice Beam in many cases."},
  {"name": "icebeam", "type": "ice", "category": "special", "power": 95, "accuracy": 100, "pp": 10,
    "desc": "A reliable special Ice-type attack with a 10% freeze chance. Excellent coverage against Grass, Ground, Flying, and Dragon types."},
  {"name": "icepunch", "type": "ice", "category": "special", "power": 75, "accuracy": 100, "pp": 15,
    "desc": "A special Ice-type attack with a 10% freeze chance. Used by Hitmonchan and Pokémon that learn it via TM."},
  {"name": "aurorabeam", "type": "ice", "category": "special", "power": 65, "accuracy": 100, "pp": 20,
    "desc": "A special Ice-type attack with a 10% chance to lower the target's Attack."},
  {"name": "iceshard", "type": "ice", "category": "special", "power": 40, "accuracy": 100, "pp": 30,
    "desc": "A weak special Ice-type attack."},
  # --- Electric-type moves ---
  {"name": "thunder", "type": "electric", "category": "special", "power": 120, "accuracy": 70, "pp": 10,
    "desc": "A very powerful special Electric-type attack with 70% accuracy and a 10% paralysis chance."},
  {"name": "thunderbolt", "type": "electric", "category": "special", "power": 95, "accuracy": 100, "pp": 15,
    "desc": "A reliable special Electric-type attack with a 10% paralysis chance. One of the best coverage moves in Gen 1."},
  {"name": "thundershock", "type": "electric", "category": "special", "power": 40, "accuracy": 100, "pp": 30,
    "desc": "A basic special Electric-type attack with a 10% paralysis chance."},
  {"name": "thunderpunch", "type": "electric", "category": "special", "power": 75, "accuracy": 100, "pp": 15,
    "desc": "A special Electric-type attack with a 10% paralysis chance."},
  {"name": "thunderwave", "type": "electric", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Paralyzes the target with 100% accuracy. Does not affect Ground types. One of the most important moves in Gen 1 because paralysis halves Speed and Speed affects critical hit rate."},
  # --- Psychic-type moves ---
  {"name": "psychic", "type": "psychic", "category": "special", "power": 90, "accuracy": 100, "pp": 10,
    "desc": "A powerful special Psychic-type attack with a 30% chance to lower the target's Special. Psychic types are dominant in Gen 1 with almost no effective counters."},
  {"name": "psybeam", "type": "psychic", "category": "special", "power": 65, "accuracy": 100, "pp": 20,
    "desc": "A special Psychic-type attack with a 10% confusion chance."},
  {"name": "confusion", "type": "psychic", "category": "special", "power": 50, "accuracy": 100, "pp": 25,
    "desc": "A special Psychic-type attack with a 10% confusion chance."},
  {"name": "dreameater", "type": "psychic", "category": "special", "power": 100, "accuracy": 100, "pp": 15,
    "desc": "A powerful special Psychic-type attack that only works on sleeping targets. Heals the user for half the damage dealt."},
  {"name": "hypnosis", "type": "psychic", "category": "status", "power": 0, "accuracy": 60, "pp": 20,
    "desc": "Puts the target to sleep. Only 60% accuracy makes it unreliable, but sleep is so powerful in Gen 1 that it's still worth using."},
  {"name": "psychup", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Copies the target's stat changes. Situationally useful."},
  {"name": "agility", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Raises the user's Speed by two stages. Also doubles the critical hit rate in Gen 1 since crits depend on Speed."},
  {"name": "amnesia", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Raises the user's Special by two stages. In Gen 1, Special governs both offensive and defensive special capability, so Amnesia doubles both your special attack power and special defense in one move. Incredibly powerful."},
  {"name": "barrier", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Raises the user's Defense by two stages. Makes the user much harder to take down physically."},
  {"name": "lightscreen", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Halves special damage taken by the user's side for 5 turns."},
  {"name": "reflect", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Halves physical damage taken by the user's side for 5 turns. In Gen 1, critical hits ignore Reflect."},
  {"name": "rest", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "The user falls asleep for two turns but fully restores HP and cures status. A powerful recovery option since sleep is the only way to heal fully in Gen 1 for many Pokémon."},
  {"name": "meditate", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Raises the user's Attack by one stage. Weaker than Swords Dance."},
  {"name": "teleport", "type": "psychic", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Flees from wild battles. Does nothing in trainer battles. Useless competitively."},
  {"name": "kinesis", "type": "psychic", "category": "status", "power": 0, "accuracy": 80, "pp": 15,
    "desc": "Lowers the target's accuracy by one stage. Exclusive to Alakazam."},
  # --- Dragon-type moves ---
  {"name": "dragonrage", "type": "dragon", "category": "special", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Always deals exactly 40 HP of damage regardless of type matchups. The only Dragon-type attack in Gen 1."},
  # --- Status / Utility moves ---
  {"name": "recover", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Restores up to 50% of the user's max HP. A crucial recovery move for Pokémon like Alakazam, Starmie, and Chansey."},
  {"name": "softboiled", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Restores up to 50% of the user's max HP. Functionally identical to Recover. Exclusive to Chansey."},
  {"name": "sing", "type": "normal", "category": "status", "power": 0, "accuracy": 55, "pp": 15,
    "desc": "Puts the target to sleep. Only 55% accuracy makes it very unreliable."},
  {"name": "lovelykiss", "type": "normal", "category": "status", "power": 0, "accuracy": 75, "pp": 10,
    "desc": "Puts the target to sleep. 75% accuracy. Exclusive to Jynx."},
  {"name": "growl", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Lowers the target's Attack by one stage."},
  {"name": "leer", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Lowers the target's Defense by one stage."},
  {"name": "tailwhip", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Lowers the target's Defense by one stage. Functionally identical to Leer."},
  {"name": "screech", "type": "normal", "category": "status", "power": 0, "accuracy": 85, "pp": 40,
    "desc": "Lowers the target's Defense by two stages. A powerful setup option for physical attackers."},
  {"name": "swordsdance", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Raises the user's Attack by two stages. The best physical boosting move. Devastating on Pokémon like Snorlax and Victreebel."},
  {"name": "growth", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Raises the user's Special by one stage. Weaker than Amnesia."},
  {"name": "defensecurl", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Raises the user's Defense by one stage."},
  {"name": "harden", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Raises the user's Defense by one stage. Functionally identical to Defense Curl."},
  {"name": "acidarmor", "type": "poison", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Raises the user's Defense by two stages. A powerful physical defensive boost."},
  {"name": "minimize", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Raises the user's evasion by one stage in Gen 1."},
  {"name": "smokescreen", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 20,
    "desc": "Lowers the target's accuracy by one stage."},
  {"name": "flash", "type": "normal", "category": "status", "power": 0, "accuracy": 70, "pp": 20,
    "desc": "Lowers the target's accuracy by one stage. Low accuracy makes it unreliable."},
  {"name": "disable", "type": "normal", "category": "status", "power": 0, "accuracy": 55, "pp": 20,
    "desc": "Disables one of the target's moves randomly for 1-8 turns. In Gen 1, which move is disabled is random."},
  {"name": "substitute", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Creates a substitute using 25% of the user's max HP. The substitute absorbs hits for the user. Very useful for blocking status and providing free turns."},
  {"name": "transform", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Transforms into the target, copying its species, stats, moves, and typing. Used by Ditto."},
  {"name": "conversion", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Changes the user's type to match one of its moves. Exclusive to Porygon."},
  {"name": "splash", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 40,
    "desc": "Does absolutely nothing. Famously useless, primarily known as Magikarp's signature move."},
  {"name": "metronome", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Randomly selects and uses any move in the game. Completely unpredictable."},
  {"name": "mimic", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "Copies one of the target's moves permanently for the rest of the battle."},
  {"name": "haze", "type": "ice", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Resets all stat changes for both sides and cures all status conditions on both sides in Gen 1. Counters setup sweepers."},
  {"name": "mist", "type": "ice", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Prevents the opponent from lowering the user's stats for 5 turns."},
  {"name": "focusenergy", "type": "normal", "category": "status", "power": 0, "accuracy": 100, "pp": 30,
    "desc": "Intended to boost critical hit rate, but in Gen 1, Focus Energy actually DIVIDES the critical hit rate by 4 due to a bug. Actively harmful to use."},
  {"name": "bide", "type": "normal", "category": "physical", "power": 0, "accuracy": 100, "pp": 10,
    "desc": "The user endures attacks for 2-3 turns, then returns double the damage received. Unreliable and rarely used."},
  {"name": "swift", "type": "normal", "category": "physical", "power": 60, "accuracy": 100, "pp": 20,
    "desc": "A special Normal-type attack that never misses. Moderate power."},
  {"name": "glare", "type": "normal", "category": "status", "power": 0, "accuracy": 75, "pp": 30,
    "desc": "Paralyzes the target. Unlike Thunder Wave, Glare can paralyze Ground types. Used by Arbok."},
  {"name": "sonicboom", "type": "normal", "category": "physical", "power": 0, "accuracy": 90, "pp": 20,
    "desc": "Always deals exactly 20 HP of damage. Weak fixed damage."},
  {"name": "constrict", "type": "normal", "category": "physical", "power": 10, "accuracy": 100, "pp": 35,
    "desc": "An extremely weak Normal-type attack with a 10% Speed drop chance."},
]


def build_species_description(species: Dict) -> str:
  """Build a rich natural-language description for a species embedding.

  The description incorporates typing, base stats, and competitive context
  to give the sentence encoder maximal semantic signal.
  """
  name = species["name"].capitalize()
  types = "/".join(t.capitalize() for t in species["types"])

  # Stat summary
  stats = species
  total = stats["hp"] + stats["atk"] + stats["def"] + stats["spa"] + stats["spd"] + stats["spe"]

  # Identify standout stats (top 2)
  stat_names = {"hp": "HP", "atk": "Attack", "def": "Defense", "spa": "Special", "spe": "Speed"}
  # In Gen 1, spa and spd are the same "Special" stat, so we just use spa
  gen1_stats = {k: v for k, v in stats.items() if k in stat_names}
  sorted_stats = sorted(gen1_stats.items(), key=lambda x: x[1], reverse=True)
  top_stats = [f"{stat_names[s[0]]} {s[1]}" for s in sorted_stats[:2]]
  low_stats = [f"{stat_names[s[0]]} {s[1]}" for s in sorted_stats[-1:]]

  description = (
    f"{name} is a {types}-type Pokémon in Generation 1. "
    f"It has base stats of {stats['hp']} HP, {stats['atk']} Attack, "
    f"{stats['def']} Defense, {stats['spa']} Special, and {stats['spe']} Speed "
    f"(total {total}). "
    f"Its strongest stats are {' and '.join(top_stats)}, "
    f"while its weakest is {', '.join(low_stats)}. "
    f"{species['desc']}"
  )
  return description


def build_move_description(move: Dict) -> str:
  """Build a rich natural-language description for a move embedding."""
  name = move["name"].replace("_", " ").title()
  mtype = move["type"].capitalize()
  category = move["category"].capitalize()

  if move["power"] > 0:
    power_str = f"It has {move['power']} base power, {move['accuracy']}% accuracy, and {move['pp']} PP."
  elif move["power"] == 0 and move["category"] != "status":
    power_str = f"It has fixed or special damage, {move['accuracy']}% accuracy, and {move['pp']} PP."
  else:
    power_str = f"It is a status move with {move['accuracy']}% accuracy and {move['pp']} PP."

  description = (
    f"{name} is a {mtype}-type {category} move in Generation 1. "
    f"{power_str} "
    f"{move['desc']}"
  )
  return description


def generate_embeddings(model_name: str = "all-mpnet-base-v2", output_dir: str = "embeddings"):
  """Generate and save all Gen 1 entity embeddings.

  Args:
    model_name: HuggingFace sentence-transformers model name.
                'all-mpnet-base-v2' produces 768-dim embeddings (recommended).
                'all-MiniLM-L6-v2' produces 384-dim (faster, smaller).
      output_dir: Directory to save embedding tensors.
  """
  from sentence_transformers import SentenceTransformer

  os.makedirs(output_dir, exist_ok=True)

  print(f"Loading sentence encoder: {model_name}")
  encoder = SentenceTransformer(model_name)
  embedding_dim = encoder.get_sentence_embedding_dimension()
  print(f"Embedding dimension: {embedding_dim}")

  # --- Species embeddings ---
  print(f"\nGenerating species embeddings for {len(GEN1_SPECIES)} Gen 1 Pokémon...")
  species_names = []
  species_texts = []
  for sp in GEN1_SPECIES:
    species_names.append(sp["name"])
    species_texts.append(build_species_description(sp))

  # Encode in batches
  species_vectors = encoder.encode(species_texts, show_progress_bar=True, convert_to_tensor=True)
  species_embeddings = {name: vec for name, vec in zip(species_names, species_vectors)}

  # Print a sample to verify
  print(f"  Sample — {species_names[0]}: shape={species_vectors[0].shape}, "
        f"norm={species_vectors[0].norm().item():.3f}")

  # --- Move embeddings ---
  print(f"\nGenerating move embeddings for {len(GEN1_MOVES)} Gen 1 moves...")
  move_names = []
  move_texts = []
  for mv in GEN1_MOVES:
    move_names.append(mv["name"])
    move_texts.append(build_move_description(mv))

  move_vectors = encoder.encode(move_texts, show_progress_bar=True, convert_to_tensor=True)
  move_embeddings = {name: vec for name, vec in zip(move_names, move_vectors)}

  print(f"  Sample — {move_names[0]}: shape={move_vectors[0].shape}, "
        f"norm={move_vectors[0].norm().item():.3f}")

  # --- Unknown token ---
  print("\nGenerating 'unknown' embedding for unrevealed Pokémon...")
  unknown_text = (
    "An unknown and unrevealed Pokémon in Generation 1. "
    "Its species, type, moves, and stats are currently unknown to the player. "
    "It could be any of the 151 Pokémon in the game. "
    "No information is available about this Pokémon yet."
  )
  unknown_vec = encoder.encode([unknown_text], convert_to_tensor=True)[0]

  # --- Build index mappings ---
  # These let the model look up embeddings by integer index during training
  species_to_idx = {name: i for i, name in enumerate(species_names)}
  move_to_idx = {name: i for i, name in enumerate(move_names)}

  embedding_index = {
    "species_names": species_names,
    "move_names": move_names,
    "species_to_idx": species_to_idx,
    "move_to_idx": move_to_idx,
    "embedding_dim": embedding_dim,
    "model_name": model_name,
  }

  # --- Save everything ---
  torch.save(species_embeddings, os.path.join(output_dir, "species_embeddings.pt"))
  torch.save(move_embeddings, os.path.join(output_dir, "move_embeddings.pt"))
  torch.save(unknown_vec, os.path.join(output_dir, "unknown_embedding.pt"))
  torch.save(embedding_index, os.path.join(output_dir, "embedding_index.pt"))

  # Also save a stacked tensor version for efficient nn.Embedding initialization
  # Species: (num_species, embedding_dim) — indexed by species_to_idx
  species_matrix = torch.stack([species_embeddings[n] for n in species_names])
  move_matrix = torch.stack([move_embeddings[n] for n in move_names])
  torch.save(species_matrix, os.path.join(output_dir, "species_matrix.pt"))
  torch.save(move_matrix, os.path.join(output_dir, "move_matrix.pt"))

  # Save metadata as JSON for easy inspection
  metadata = {
    "generated_at": datetime.now().isoformat(),
    "model_name": model_name,
    "embedding_dim": embedding_dim,
    "num_species": len(species_names),
    "num_moves": len(move_names),
    "species_list": species_names,
    "move_list": move_names,
    "generation": 1,
    "notes": (
      "Gen 1 has no abilities. Items are not used in gen1randombattle format. "
      "In Gen 1, the Special stat governs both SpA and SpD. "
      "Move categories are determined by type, not by the move itself."
    ),
  }
  with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

  print(f"\n{'='*60}")
  print(f"Embeddings saved to: {output_dir}/")
  print(f"  species_embeddings.pt  — {len(species_names)} species (dict)")
  print(f"  species_matrix.pt      — ({len(species_names)}, {embedding_dim}) tensor")
  print(f"  move_embeddings.pt     — {len(move_names)} moves (dict)")
  print(f"  move_matrix.pt         — ({len(move_names)}, {embedding_dim}) tensor")
  print(f"  unknown_embedding.pt   — ({embedding_dim},) tensor")
  print(f"  embedding_index.pt     — name↔index mappings")
  print(f"  metadata.json          — generation metadata")
  print(f"{'='*60}")

  return species_embeddings, move_embeddings, unknown_vec, embedding_index


def verify_embeddings(output_dir: str = "embeddings"):
    """Quick verification: load saved embeddings and check similarity."""
    print("\n--- Verification: loading saved embeddings ---")

    species_emb = torch.load(os.path.join(output_dir, "species_embeddings.pt"))
    move_emb = torch.load(os.path.join(output_dir, "move_embeddings.pt"))
    index = torch.load(os.path.join(output_dir, "embedding_index.pt"))

    print(f"Species count: {len(species_emb)}")
    print(f"Move count: {len(move_emb)}")
    print(f"Embedding dim: {index['embedding_dim']}")

    # Cosine similarity between related Pokémon
    from torch.nn.functional import cosine_similarity

    pairs_to_check = [
        ("charizard", "arcanine", "Both Fire types, should be similar"),
        ("alakazam", "starmie", "Both top-tier Psychic special attackers"),
        ("snorlax", "chansey", "Both bulky Normal types"),
        ("charizard", "geodude", "Fire vs Rock/Ground, should differ"),
        ("mewtwo", "magikarp", "Strongest vs weakest, should differ"),
    ]

    print("\nSpecies cosine similarities:")
    for name_a, name_b, reason in pairs_to_check:
        sim = cosine_similarity(
            species_emb[name_a].unsqueeze(0),
            species_emb[name_b].unsqueeze(0)
        ).item()
        print(f"  {name_a:12s} ↔ {name_b:12s} = {sim:+.3f}  ({reason})")

    # Move similarities
    move_pairs = [
        ("thunderbolt", "thunder", "Same type, different power/accuracy"),
        ("surf", "icebeam", "Common special coverage pair"),
        ("sleeppowder", "spore", "Both sleep moves"),
        ("earthquake", "psychic", "Physical ground vs special psychic"),
        ("swordsdance", "amnesia", "Both +2 boost moves"),
    ]

    print("\nMove cosine similarities:")
    for name_a, name_b, reason in move_pairs:
        sim = cosine_similarity(
            move_emb[name_a].unsqueeze(0),
            move_emb[name_b].unsqueeze(0)
        ).item()
        print(f"  {name_a:14s} ↔ {name_b:14s} = {sim:+.3f}  ({reason})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gen 1 Pokémon LLM embeddings")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Sentence-transformers model name (default: all-mpnet-base-v2)")
    parser.add_argument("--output-dir", default="embeddings",
                        help="Output directory (default: embeddings/)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after generation")
    args = parser.parse_args()

    generate_embeddings(model_name=args.model, output_dir=args.output_dir)

    if args.verify:
        verify_embeddings(output_dir=args.output_dir)