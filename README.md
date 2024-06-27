## Core LLama2 Math Functions in OpenCL 

This repo shows a PoC with the core math functions in Llama2.c written for OpenCL. 
This just shows a quick demo of how those kernels could be written in OpenCL to be 
dispatched on a compatible OpenCL device. 

### How to build?

```bash
mkdir build
cd build
cmake ..
make 
```

### How to run? 

```bash
./corellms
```

### License 

[MIT](LICENSE)