test: superres.cpp
	g++ -o superres superres.cpp `pkg-config --cflags --libs opencv`
