# Realtime Video Stablization
## Naive Implementation
under ```./naive```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Not realtime
= Using moving average to smooth
## Kalman Filter Implementation
under ```./kalmanFilter```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Realtime
- Using kalman filter to smooth
## Improved Kalman Filter Implementation
under ```./kalmanfilterImproved```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Using the transformation on the current frame. 
(The original version uses the previous frame, which introduces latency by 1 frame)
- Reset the kalman filter when large movement detected. 
The goal is to improve the performance during desired camera movement.