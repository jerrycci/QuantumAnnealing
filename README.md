# Requirements
OS requirement : 
1. Ubuntu == 20.04
2. Nvidia Driver Version == 495.29.05
3. CUDA versuib == 11.5

Python Env requirement :
1. python == 3.9
2. pyqubo == 1.2.0
3. numpy == 1.23.3
4. tqdm == 4.64.1

# Build Project
1. Make sure your host env as same as OS requirement
2. Download Dockerfile
3. Build Docker Image
   ```
   sudo docker build -t gpuda. --no-cache
   ```
4. Docker Run gpuda
   ```
   sudo docker run -it -p 8080:22 --gpus all gpuda bash
   ```

