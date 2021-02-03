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

Some functions use Cython code, which needs to be compiled. Just run the `1. Synchronize good stations.ipynb` notebook first. Cython functions use 4 CPU cores by default, adjust `N_THREADS` variable in `optimize.pyx` file.  

#### Data sets

All data files should be in ./data folder. Only three files were used in this solution:
 - round2_competition.csv
 - round2_sensors.csv
 - round2_training1.csv


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
## 2.1 Wave velocity model

In round 1 of the competition many participants used effective wave velocity instead of speed of light to estimate distance by time-of-flight. @richardalligier found its value using optimization technique. In round 2 I improved this model by introducing altitude dependence of wave velocity. 

Using altitude dependence of refractive index <img src="svgs/0b700b6ef9752b739fe4ee8dc2925d28.svg?invert_in_darkmode" align=middle width=32.12352pt height=24.65759999999998pt/> from [1], velocity as a function of altitude can be written as follows: 

<p align="center"><img src="svgs/78ba6690fac5dc48d1b3aacfade2f2f3.svg?invert_in_darkmode" align=middle width=213.41924999999998pt height=33.583769999999994pt/></p>

, where <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.113876000000004pt height=14.155350000000013pt/> is the speed of light, <img src="svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.471165000000003pt height=22.831379999999992pt/> - altitude, <img src="svgs/0b700b6ef9752b739fe4ee8dc2925d28.svg?invert_in_darkmode" align=middle width=32.12352pt height=24.65759999999998pt/> - refractive index, <img src="svgs/2e5cace905a61fe431f7b898becb0be1.svg?invert_in_darkmode" align=middle width=18.881445000000006pt height=22.46574pt/> and <img src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.293555000000003pt height=22.46574pt/> - some constants.

Instead of integrating velocity each time, let's consider some effective velocity: 
<p align="center"><img src="svgs/324d302c449c8b7a25e54fbe21a471f8.svg?invert_in_darkmode" align=middle width=143.40314999999998pt height=38.810145pt/></p>

<p align="center"><img src="svgs/7668dde8336ca86314c642afcfb541ab.svg?invert_in_darkmode" align=middle width=752.4626999999999pt height=42.92277pt/></p>

, where <img src="svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.187330000000003pt height=22.46574pt/> - wave path, <img src="svgs/5a95dbebd5e79e850a576db54f501ab8.svg?invert_in_darkmode" align=middle width=16.023645000000005pt height=22.831379999999992pt/> and <img src="svgs/0f7cea0b89929faf20eda59174bc247f.svg?invert_in_darkmode" align=middle width=16.023645000000005pt height=22.831379999999992pt/> - initial and final altitudes of the wave path.

Finally, after inserting <img src="svgs/09167708b6619ba3b4d57545fe6e5937.svg?invert_in_darkmode" align=middle width=73.57482pt height=29.461410000000004pt/>, effective wave velocity will be:
<p align="center"><img src="svgs/cf06bd3cc64a15d142e37f237399e5e1.svg?invert_in_darkmode" align=middle width=319.3311pt height=39.53796pt/></p>

New wave velocity model shows 0.1m less average residual error in solving multilateral equations for 35 good stations (see `1. Synchronize good stations.ipynb` notebook).

[1] R. Purvinskis et al. Multiple Wavelength Free-Space Laser Communications. Proceedings of SPIE - The International Society for Optical Engineering, 2003. 

## 2.2 Stations time drift

Stations are synchronized when there is no time drift, so measured time is equal to aircraft time + time-of-flight:

<p align="center"><img src="svgs/43580024997948d3e68c2ff8a486aa37.svg?invert_in_darkmode" align=middle width=150.28794pt height=33.629475pt/></p>

If station measurements have a drift, then:

<p align="center"><img src="svgs/2550b51fa5c8c827afe0d44672968a2d.svg?invert_in_darkmode" align=middle width=315.3315pt height=33.629475pt/></p>
It's worth to notice here that drift is added at the moment of wave detection!

We have to have some already synchronized stations. Let's consider a synchronized station 1 and a drifted station 2.

<p align="center"><img src="svgs/1eed11d33672dd90ecde0d32e49238d6.svg?invert_in_darkmode" align=middle width=241.69529999999997pt height=33.629475pt/></p>

Considering <img src="svgs/31d18a2424dd7476a46822fd19f48a1b.svg?invert_in_darkmode" align=middle width=135.345375pt height=31.780980000000003pt/> and inserting corresponding equation for station 1, we get the resulting formula:

<p align="center"><img src="svgs/d308ef49eaec380cebfc9bf6d2da5414.svg?invert_in_darkmode" align=middle width=279.92085pt height=33.629475pt/></p>

### Time drift approximation
Drift was approximated by a sum of a linear function and a spline:
<p align="center"><img src="svgs/40d75a8025d335645062e323b7d5e5ea.svg?invert_in_darkmode" align=middle width=225.20685pt height=16.438356pt/></p>

So,
<p align="center"><img src="svgs/e1dc6ec661976b0794dd68ee39114674.svg?invert_in_darkmode" align=middle width=504.68385pt height=33.629475pt/></p>
<p align="center"><img src="svgs/0a8b9ea411938f2f635b8208b0cdaafb.svg?invert_in_darkmode" align=middle width=433.3411499999999pt height=33.629475pt/></p>

It would be very difficult to solve the last nonlinear equation directly. Instead, we will use the fact that spline eliminates the slow component of time drift and therefore in the first approximation we can simply ignore it:

<p align="center"><img src="svgs/308212430e2ac77582e93a3aed44a2fa.svg?invert_in_darkmode" align=middle width=152.39399999999998pt height=34.999305pt/></p>

Finally, we can synchronize station measurements by applying the following trasformation to measured time values:
<p align="center"><img src="svgs/43ff937d55ed29e1006238fd7c4df947.svg?invert_in_darkmode" align=middle width=378.873pt height=41.067015pt/></p>


# 3. Solution

## 3.1 Compute parameters of signal velocity model, sensors positions and time shifts for 35 good stations: `1. Synchronize good stations.ipynb`

Here we select 35 best 'good' stations out of 45 marked. A good station shouldn't have visible time drift and should have pairs with several other good stations (we should be able to optimize its location).

For 35 selected stations a subset of points (20,000 per station) was prepared to reduce computation complexity. On this subset average L1 loss <img src="svgs/5d6189b601b6b15604e05866ec8efa5c.svg?invert_in_darkmode" align=middle width=331.86565499999995pt height=31.780980000000003pt/> was minimized. <img src="svgs/929ed909014029a206f344a28aa47d15.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/> and <img src="svgs/4327ea69d9c5edcc8ddaf24f1d5b47e4.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/> are distances from aircraft to stations, <img src="svgs/ed3d6a7ea65a223451a604b6372c870a.svg?invert_in_darkmode" align=middle width=37.15305pt height=31.780980000000003pt/> and <img src="svgs/87d5c3931435576d25da229aa5fbd5f3.svg?invert_in_darkmode" align=middle width=37.15305pt height=31.780980000000003pt/> are constant stations time shifts.


## 3.2 Add station 150 using training1 dataset `2. Add station 150.ipynb`

## 3.3 Synchronize all stations

## 3.4 Predict and filter tracks


# Notes
