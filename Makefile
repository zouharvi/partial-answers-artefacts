clean:
	rm -rf data/final

assert_top:
	@echo "Make sure you're running me from the top-level directory, so that data will be stored in 'data/final/' [Ctrl-C to exit, ENTER to continue]"
	@read _

_data:
	mkdir -p data/final/
	wget --user finalproject --password Rik@LfD21 -P data/final/ https://teaching.stijneikelboom.nl/lfd2122/COP.filt3.sub.zip
	unzip -j -d data/final/ data/final/COP.filt3.sub.zip

	# join all JSON files together in a hacky way
	echo "[" > data/final/all.json
	for f in data/final/COP*.filt3.sub.json ; do \
		cat $$f ; \
		echo -n "," ; \
	done >> data/final/all.json;
	echo "]" >> data/final/all.json

	# remove trailing comma
	sed 's/,]/]/' -i data/final/all.json

	# remove temporary files
	rm data/final/COP*.filt3.sub.json
	rm data/final/COP.filt3.sub.zip

data: assert_top _data

data_all: assert_top _data prepare_data craft_data

prepare_data:
	python3 ./src/prepare_data.py --data-in data/final/all.json --data-out data/final/clean.json

craft_data:
	python3 ./src/craft_data.py --data-in data/final/clean.json --data-out data/final/{LABEL}.json

get_glove:
	wget -P data/ http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
	unzip -d data/ data/glove.6B.zip
	python3 ./src/prepare_glove.py --data-in data/glove.6B.200d.txt --data-out data/glove.pkl