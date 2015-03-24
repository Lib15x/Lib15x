CC_INCLUDE = -I/opt/eigen-3.2.4/
LD_FLAGS =  -lrt

TARGETS = Main

all: $(TARGETS)

Main: Main.cpp ${wildcard *.hpp}
	g++ $< -o $@ -O3 $(CC_INCLUDE) $(LD_FLAGS) -Wall -std=c++1y

clean:
	rm -f $(TARGETS)

again: clean $(TARGETS)
