CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`

% : %.cu
	nvcc $(CFLAGS) $(LIBS) -o $@ $<