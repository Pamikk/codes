CC = g++
CFLAGS = -g -Wall
SRCS = src/load.cpp
PROG = load

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) -I /usr/local/include/opencv4 $(SRCS) $(LIBS)