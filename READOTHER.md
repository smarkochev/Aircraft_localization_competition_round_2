# 1. Setup

#### To setup python environment:
```
>> conda env create --file environment.yml
```
#### To run notebooks:
```
>> conda activate gt
```
```
(gt) >> jupyter notebook
```

Some functions use Cython code, which needs to be compiled. Just run the `1. Synchronize good stations.ipynb` notebook first. Cython functions use 4 CPU cores by default, adjust `NTHREADS` variable in `optimize.pyx` file.  

#### Data sets

All data files should be in ./data folder. Only three files were used in this solution:
 - round2competition.csv
 - round2sensors.csv
 - round2training1.csv


# /src files description

 | Filename     |  Description  |
 |--------------|---------------|
 | filters.py   | median and graph filters |
 | geo.py       | functions to calculate distance, effective velocity; to transform coordinates and to plot altitude profiles |
 | optimize.pyx | collection of cython functions to solve multelateral equations efficiently |
 | solvers.py   | GoodStationsSolver and SingleStationSolver classes |
 | stations.py  | Stations class including time correction method |
 | track.py     | Track and TrackCollection classes to work with tracks |


# 2. Theory
## 2.1 Signal velocity model

In round 1 of the competition many participants used effective signal velocity instead of speed of light to estimate distance by time-of-flight. @richardalligier found its value using optimization technique. In round 2 I improved this model by introducing altitude dependence of signal velocity. 

Using altitude dependence of refractive index $n(h)$ from [1], velocity as a function of altitude can be written as follows: 

$$v(h) = \frac{c}{n(h)} = \frac{c}{1 + A_0\cdot e^{-B\cdot h}}$$
, where $c$ is the speed of light, $h$ - altitude, $n(h)$ - refractive index, $A_0$ and $B$ - some constants.

Instead of integrating velocity each time, let's consider some constant effective velocity: 
$$\hat{v} = const = \frac{L}{\int{dt(h)}}$$

$$\int{dt(h)} = \int_0^L{\frac{1+A_0\cdot e^{-B\cdot l\cdot \sin{\phi}}}{c}\cdot dl} = \int_{h_1}^{h_2}{\frac{1+A_0\cdot e^{-B\cdot h}}{c\cdot \sin{\phi}}\cdot dh} = \frac{h_2 - h_1}{c\cdot \sin{\phi}} + \frac{A_0}{c\cdot B\cdot \sin{\phi}}\cdot(e^{-B\cdot h_1} - e^{-B\cdot h_2})$$
, where $L$ - signal path, $h_1$ and $h_2$ - initial and final altitudes of the signal path.

Finally, after inserting $L = \frac{h_2 - h_1}{\sin{\phi}}$, effective signal velocity will be:
$$\hat{v} = \frac{c}{1+\frac{A_0}{B\cdot (h_2 - h_1)}(e^{-B\cdot h_1}-e^{-B\cdot h_2})}, h_2 > h_1$$

New signal velocity model shows 0.1m less average residual error in solving multilateral equations for 35 good stations (see `1. Synchronize good stations.ipynb` notebook).

[1] R. Purvinskis et al. Multiple Wavelength Free-Space Laser Communications. Proceedings of SPIE - The International Society for Optical Engineering, 2003. 

## 2.2 Stations time drift

Stations are synchronized when there is no time drift, so measured time is equal to aircraft time + time-of-flight:

$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}}$$

If station measurements have a drift, then:

$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}} + drift(t^{aircraft} + \frac{L}{\hat{v}})$$
It's worth to notice here that drift is added at the moment of signal detection!

We have to have some already synchronized stations. Let's consider a synchronized station 1 and a drifted station 2.

$$drift(t_2) = t_2^{meas} - t_2^{aircraft} - \frac{L_2}{\hat{v}}$$

Considering $t_2^{aircraft} \triangleq t_1^{aircraft}$ and inserting corresponding equation for station 1, we get the resulting formula:

$$drift(t_2) = t_2^{meas} - \frac{L_2}{\hat{v}} - (t_1^{meas} - \frac{L_1}{\hat{v}})$$

### Time drift approximation
Drift was approximated by a sum of a linear function and a spline:
$$drift(t) = A\cdot t + B + spline(t)$$

So,
$$t^{meas} = t^{aircraft} + \frac{L}{\hat{v}} + A\cdot(t^{aircraft} + \frac{L}{\hat{v}}) + B + spline(t^{aircraft} + \frac{L}{\hat{v}})$$
$$t^{meas} = (A+1)\cdot(t^{aircraft} + \frac{L}{\hat{v}}) + B + spline(t^{aircraft} + \frac{L}{\hat{v}})$$

It would be very difficult to solve the last nonlinear equation directly. Instead, we will use the fact that spline eliminates the slow component of time drift and therefore in the first approximation we can simply ignore it:

$$t^{aircraft} + \frac{L}{\hat{v}} = \frac{t^{meas} - B}{A+1}$$

Finally, we can synchronize station measurements by applying the following trasformation to measured time values:
$$t^{aircraft} + \frac{L}{\hat{v}} \triangleq t^{sync} = \frac{t^{meas} - B - spline(\frac{t^{meas} - B}{A+1})}{A+1}$$


# 3. Solution

## 3.1 Compute parameters of signal velocity model, sensors positions and time shifts for 35 good stations: `1. Synchronize good stations.ipynb`

Here we select 35 best 'good' stations out of 45 marked. A good station shouldn't have visible time drift and should have pairs with several other good stations (we should be able to optimize its location).

For 35 selected stations a subset of points (20,000 per station) was prepared to reduce computation complexity. On this subset average L1 loss $|L_1 - L_2 - \hat{v}\cdot(t_1^{meas} + t_1^{shift} - t_2^{meas} - t_2^{shift})|$ was minimized. $L_1$ and $L_2$ are distances from aircraft to stations, $t_1^{shift}$ and $t_2^{shift}$ are constant stations time shifts.


## 3.2 Add station 150 using training1 dataset `2. Add station 150.ipynb`

## 3.3 Synchronize all stations

## 3.4 Predict and filter tracks


# Notes