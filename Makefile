
TARGET:= hiccups
CXXFLAGS:= -O2 -W -Wall

all: $(TARGET)

hiccups: hiccups.cpp
	g++ -pthread $(CXXFLAGS) -o hiccups hiccups.cpp -lpthread

clean:
	rm -f *.o *~ $(TARGET)
