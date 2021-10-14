#!/usr/bin/env bash

echo "Make sure you're running me from the top-level directory, so that data will be stored in 'data/final/' [Ctrl-C to exit]"
read _

mkdir -p data/final/
wget --user finalproject --password Rik@LfD21 -P data/final/ https://teaching.stijneikelboom.nl/lfd2122/COP.filt3.sub.zip
unzip -j -d data/final/ data/final/COP.filt3.sub.zip

# join all JSON files together in a hacky way
echo "[" > data/final/COP.all.json
for f in data/final/COP*.filt3.sub.json; do
	 cat $f;
	 echo -n ","
done >> data/final/COP.all.json
echo "]" >> data/final/COP.all.json

# remove trailing comma
sed 's/,]/]/' -i data/final/COP.all.json