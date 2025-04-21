# Dynamic Array in CUDA

In reality, all multidimensional array in C are liniearized, this is due to use of a
"flat" memory space in modern computers, whereas each of the memory byte is labelled with
an address that ranges from 0 to the largest number used.

the current CUDA C compiler leaves the work of liniearize the multidimensional array to the programmer.
because us, programmers, know our data and memory access pattern better than compiler does.
