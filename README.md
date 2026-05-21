# number-recognition
# Notice: I will update this file later if i really free.

## Folder structure
```
root/
├── data/
├── include/
│   ├── matrix.h
│   ├── dataloader.h
│   └── network.h    
├── src/      
│   ├── matrix.cpp
│   ├── dataloader.cpp
│   └── main.cpp
└── Makefile
```


Build
```batch
g++ -O3 -std=c++17 -fopenmp -I./include src/*.cpp -o mnist_nn 
mnist_nn
```
