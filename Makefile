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
	echo "[" > data/final/COP.all.json
	for f in data/final/COP*.filt3.sub.json ; do \
		cat $$f ; \
		echo -n "," ; \
	done >> data/final/COP.all.json;
	echo "]" >> data/final/COP.all.json

	# remove trailing comma
	sed 's/,]/]/' -i data/final/COP.all.json

data: assert_top _data

data_all: assert_top _data
	python3 ./src/prepare_data.py --data-in data/final/COP.all.json --data-out data/final/COP.cleaned.json