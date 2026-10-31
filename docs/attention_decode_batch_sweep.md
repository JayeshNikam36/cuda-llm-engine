#0 building with "default" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 239B 0.0s done
#1 DONE 0.1s

#2 [internal] load metadata for docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
#2 DONE 0.7s

#3 [internal] load .dockerignore
#3 transferring context: 2B 0.1s done
#3 DONE 0.1s

#4 [1/3] FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5
#4 resolve docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5 0.0s done
#4 DONE 0.0s

#5 [2/3] RUN apt-get update && apt-get install -y     cmake ninja-build git python3 python3-venv python3-pip     && rm -rf /var/lib/apt/lists/*
#5 CACHED

#6 [3/3] WORKDIR /workspace
#6 CACHED

#7 exporting to image
#7 exporting layers done
#7 exporting manifest sha256:964021ec8d73946ad926ffd5353305924f1b7b18883f92d7418fc576e0c180b0 done
#7 exporting config sha256:8f47857d6d257accffd25b8a28f614a6dc535348b6f3e7212bea862496079965 done
#7 exporting attestation manifest sha256:37f8c1f3896e246df0750b36e69ea8078728bd576ace73062f403f545ac60a56 0.0s done
#7 exporting manifest list sha256:42b8c04a79fb3225e41f41a3903da00c08c1bb35076027b72ddb82467ce5fd54 0.0s done
#7 naming to docker.io/library/cuda-engine-dev:latest done
#7 unpacking to docker.io/library/cuda-engine-dev:latest 0.0s done
#7 DONE 0.1s
# Decode attention: naive paged kernel vs fast split-KV kernel (v1)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 1 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 1 sequences.
| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 61.4 | 8.5 | 4.9% | 1.0x |
| 256 | fast v1 | 1 | 20.5 | 25.6 | 14.6% | 3.0x |
| 256 | fast v1 | 2 | 15.4 | 34.0 | 19.4% | 4.0x |
| 256 | fast v1 | 4 | 15.4 | 34.1 | 19.5% | 4.0x |
| 256 | fast v1 | 8 | 15.4 | 34.1 | 19.5% | 4.0x |
| 256 | fast v1 | 16 | 17.4 | 30.1 | 17.2% | 3.5x |
| 256 | fast v1 | 32 | 18.4 | 28.4 | 16.3% | 3.3x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 220.2 | 9.5 | 5.4% | 1.0x |
| 1024 | fast v1 | 1 | 51.2 | 41.0 | 23.4% | 4.3x |
| 1024 | fast v1 | 2 | 35.8 | 58.5 | 33.4% | 6.1x |
| 1024 | fast v1 | 4 | 32.8 | 64.0 | 36.6% | 6.7x |
| 1024 | fast v1 | 8 | 26.6 | 78.8 | 45.0% | 8.3x |
| 1024 | fast v1 | 16 | 29.7 | 70.6 | 40.4% | 7.4x |
| 1024 | fast v1 | 32 | 31.7 | 66.1 | 37.8% | 6.9x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 361.5 | 11.6 | 6.6% | 1.0x |
| 2048 | fast v1 | 1 | 82.9 | 50.6 | 28.9% | 4.4x |
| 2048 | fast v1 | 2 | 58.4 | 71.9 | 41.1% | 6.2x |
| 2048 | fast v1 | 4 | 50.2 | 83.6 | 47.8% | 7.2x |
| 2048 | fast v1 | 8 | 41.0 | 102.4 | 58.5% | 8.8x |
| 2048 | fast v1 | 16 | 43.0 | 97.5 | 55.7% | 8.4x |
| 2048 | fast v1 | 32 | 45.1 | 93.1 | 53.2% | 8.0x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |
# Decode attention: naive paged kernel vs fast split-KV kernel (v1)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 2 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 2 sequences.
| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 51.2 | 10.2 | 5.9% | 1.0x |
| 256 | fast v1 | 1 | 10.2 | 51.2 | 29.3% | 5.0x |
| 256 | fast v1 | 2 | 9.7 | 53.9 | 30.8% | 5.3x |
| 256 | fast v1 | 4 | 8.7 | 60.2 | 34.4% | 5.9x |
| 256 | fast v1 | 8 | 9.7 | 53.9 | 30.8% | 5.3x |
| 256 | fast v1 | 16 | 10.8 | 48.8 | 27.9% | 4.8x |
| 256 | fast v1 | 32 | 12.3 | 42.7 | 24.4% | 4.2x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 179.2 | 11.7 | 6.7% | 1.0x |
| 1024 | fast v1 | 1 | 29.2 | 71.9 | 41.1% | 6.1x |
| 1024 | fast v1 | 2 | 24.6 | 85.3 | 48.8% | 7.3x |
| 1024 | fast v1 | 4 | 20.5 | 102.4 | 58.5% | 8.7x |
| 1024 | fast v1 | 8 | 21.5 | 97.5 | 55.7% | 8.3x |
| 1024 | fast v1 | 16 | 23.0 | 91.0 | 52.0% | 7.8x |
| 1024 | fast v1 | 32 | 23.0 | 91.0 | 52.0% | 7.8x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 353.8 | 11.9 | 6.8% | 1.0x |
| 2048 | fast v1 | 1 | 49.7 | 84.5 | 48.3% | 7.1x |
| 2048 | fast v1 | 2 | 43.0 | 97.5 | 55.7% | 8.2x |
| 2048 | fast v1 | 4 | 36.4 | 115.4 | 65.9% | 9.7x |
| 2048 | fast v1 | 8 | 37.4 | 112.2 | 64.1% | 9.5x |
| 2048 | fast v1 | 16 | 39.4 | 106.4 | 60.8% | 9.0x |
| 2048 | fast v1 | 32 | 38.9 | 107.8 | 61.6% | 9.1x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |
# Decode attention: naive paged kernel vs fast split-KV kernel (v1)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 4 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 4 sequences.
| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 49.4 | 10.6 | 6.1% | 1.0x |
| 256 | fast v1 | 1 | 7.2 | 73.1 | 41.8% | 6.9x |
| 256 | fast v1 | 2 | 6.1 | 85.3 | 48.8% | 8.0x |
| 256 | fast v1 | 4 | 6.9 | 75.9 | 43.3% | 7.1x |
| 256 | fast v1 | 8 | 7.4 | 70.6 | 40.4% | 6.7x |
| 256 | fast v1 | 16 | 8.7 | 60.2 | 34.4% | 5.7x |
| 256 | fast v1 | 32 | 9.7 | 53.9 | 30.8% | 5.1x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 176.6 | 11.9 | 6.8% | 1.0x |
| 1024 | fast v1 | 1 | 21.5 | 97.5 | 55.7% | 8.2x |
| 1024 | fast v1 | 2 | 18.2 | 115.4 | 65.9% | 9.7x |
| 1024 | fast v1 | 4 | 18.7 | 112.2 | 64.1% | 9.5x |
| 1024 | fast v1 | 8 | 20.0 | 105.0 | 60.0% | 8.8x |
| 1024 | fast v1 | 16 | 19.2 | 109.2 | 62.4% | 9.2x |
| 1024 | fast v1 | 32 | 20.0 | 104.8 | 59.9% | 8.8x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 352.5 | 11.9 | 6.8% | 1.0x |
| 2048 | fast v1 | 1 | 35.6 | 117.9 | 67.4% | 9.9x |
| 2048 | fast v1 | 2 | 34.0 | 123.2 | 70.4% | 10.4x |
| 2048 | fast v1 | 4 | 34.3 | 122.3 | 69.9% | 10.3x |
| 2048 | fast v1 | 8 | 35.1 | 119.6 | 68.3% | 10.1x |
| 2048 | fast v1 | 16 | 34.8 | 120.5 | 68.8% | 10.1x |
| 2048 | fast v1 | 32 | 35.1 | 119.6 | 68.3% | 10.1x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |
# Decode attention: naive paged kernel vs fast split-KV kernel (v1)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 16 sequences.
| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 47.0 | 11.2 | 6.4% | 1.0x |
| 256 | fast v1 | 1 | 4.7 | 112.0 | 64.0% | 10.0x |
| 256 | fast v1 | 2 | 5.1 | 102.4 | 58.5% | 9.2x |
| 256 | fast v1 | 4 | 5.0 | 105.0 | 60.0% | 9.4x |
| 256 | fast v1 | 8 | 5.2 | 99.9 | 57.1% | 9.0x |
| 256 | fast v1 | 16 | 6.5 | 81.1 | 46.3% | 7.3x |
| 256 | fast v1 | 32 | 7.5 | 70.0 | 40.0% | 6.3x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 175.4 | 12.0 | 6.8% | 1.0x |
| 1024 | fast v1 | 1 | 16.6 | 126.5 | 72.3% | 10.6x |
| 1024 | fast v1 | 2 | 17.0 | 123.7 | 70.7% | 10.3x |
| 1024 | fast v1 | 4 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 8 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 16 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 32 | 18.2 | 115.0 | 65.7% | 9.6x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 350.1 | 12.0 | 6.8% | 1.0x |
| 2048 | fast v1 | 1 | 29.1 | 144.4 | 82.5% | 12.1x |
| 2048 | fast v1 | 2 | 29.8 | 140.9 | 80.5% | 11.8x |
| 2048 | fast v1 | 4 | 29.2 | 143.4 | 81.9% | 12.0x |
| 2048 | fast v1 | 8 | 29.5 | 142.2 | 81.2% | 11.9x |
| 2048 | fast v1 | 16 | 30.2 | 138.8 | 79.3% | 11.6x |
| 2048 | fast v1 | 32 | 31.4 | 133.7 | 76.4% | 11.2x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |
