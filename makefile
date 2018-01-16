.PHONY: clean build develop install
rebuild: clean build install

clean:
	rm -rf cpp/build

build: 
	mkdir -p cpp/build
	cd cpp/build && cmake ..
	cd cpp/build && make

develop: build
	python setup.py clean --all develop

install: build
	python setup.py clean --all install

