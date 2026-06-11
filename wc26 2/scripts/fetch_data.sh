#!/usr/bin/env bash
# Refresh vendored raw data from upstream sources.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)

git clone --depth 1 https://github.com/martj42/international_results "$TMP/ir"
cp "$TMP/ir/results.csv" "$TMP/ir/former_names.csv" data/raw/international_results/

git clone --depth 1 https://github.com/jfjelstul/worldcup "$TMP/wc"
cp "$TMP/wc/data-csv/matches.csv" data/raw/worldcup/

git clone --depth 1 https://github.com/Dato-Futbol/fifa-ranking "$TMP/fr"
cp "$TMP/fr/ranking_fifa_historical.csv" data/raw/fifa_ranking/

rm -rf "$TMP"
echo "raw data refreshed"
