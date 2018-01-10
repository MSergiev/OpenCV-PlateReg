lpreg: superres.cpp
	g++ -o lpreg superres.cpp `pkg-config --cflags --libs opencv`
