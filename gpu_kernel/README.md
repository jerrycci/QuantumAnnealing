# CudaDigitalAnnealing

[Code Source](https://github.com/Shutoparu/CudaDigitalAnnealing.git)

Overall time complexity: O(sweep*dim)

## Hardware and docker image information

Used OS: Ubuntu 20.04.3 LTS

kernel: Linux 5.13.0-51-generic

Used image version: nvidia/cuda:10.0-devel-ubuntu18.04

command used to create container: 
```
docker run -it --gpus all nvidia/cuda:10.0-devel-ubuntu18.04 bash
```
## To use the algorithm:

First of all, generate a cudaDA.so file for the cudaDigitalAnnealing.cu file.

In command line, run the following prompt:
```
nvcc --compiler-options -fPIC -shared -arch sm_70 --maxrregcount=255 -o ./cudaDA.so cudaDigitalAnnealing.cu
```
Locate the cudaDA.so file and main.py file, put them in the same folder.

Second, import the algorithm to your code:
```
from main import DA
```
Then you can use the algorithm with following code:
```
algorithm = DA()
algorithm.run()
```
### result:
For a ising problem with 727 binary bits, the algorithm spent 10.5 seconds to finish 100,000 iterations to find a local minima of the problem.
![result](imagesesult.png)

### compare to SA: 
the prototype of the algorithm spent 59 seconds to finish 100,000 iterations,
compared to python Simulated Annealing algorithm which spent 200 seconds