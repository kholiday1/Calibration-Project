# Quick Start

1. Clone this code with `git clone https://gitlab.com/marinesitu-public/calibration-development.git`

2. Create a virtual environment with `python3 -m venv venv`

3. Install the reqs with `pip3 install -r requirements.txt`

4. Download the code from here (https://drive.google.com/drive/u/0/folders/1Z4CPUncCtTOB02gYyg1_i3JdFnhXwduA) to somewhere on your machine. 

5. Change lines 160 and 161 (variables LEFT_FOLDER and RIGHT_FOLDER) to point to this downloaded data.

6. Run the calibrate script with `python3 calibrate_data.py`. It should output something like 
```
Stereo Calibration Complete!

--- Calibration Results ---
Left Camera Matrix (mtxL):
 [[1.06920884e+03 0.00000000e+00 6.18251141e+02]
 [0.00000000e+00 1.06897199e+03 4.04928481e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Left Distortion Coeffs (distL):
 [[ 6.58269984e-01 -2.12274776e+00  4.67000007e-03 -1.56201161e-03
   4.76380443e+00]]
Right Camera Matrix (mtxR):
 [[1.06757374e+03 0.00000000e+00 6.20886169e+02]
 [0.00000000e+00 1.06882263e+03 3.95953039e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Right Distortion Coeffs (distR):
 [[ 7.44799047e-01 -3.27011493e+00  4.32894773e-03 -1.68361524e-03
   8.76605648e+00]]
Rotation Matrix (R):
 [[ 9.99995264e-01  3.07708419e-03  5.47355770e-05]
 [-3.07706472e-03  9.99995204e-01 -3.52383894e-04]
 [-5.58196294e-05  3.52213800e-04  9.99999936e-01]]
Translation Vector (T):
 [[-0.07496879]
 [-0.00015043]
 [-0.0003273 ]]
Essential Matrix (E):
 [[-9.98726549e-07  3.27245479e-04 -1.50543234e-04]
 [-3.31483212e-04  2.53979120e-05  7.49687655e-02]
 [ 3.81111009e-04 -7.49679658e-02  2.64260272e-05]]
Fundamental Matrix (F):
 [[-1.67230528e-09  5.48073546e-07 -4.90417772e-04]
 [-5.54399398e-07  4.24869391e-08  1.34387066e-01]
 [ 9.01822994e-04 -1.34398465e-01  1.00000000e+00]]
```