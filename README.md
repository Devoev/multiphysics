This repo contains the code for the TU Darmstadt lecture [18-sc-2030 - Simulation multiphysikalischer Probleme](https://www.tucan.tu-darmstadt.de/scripts/mgrqispi.dll?APPNAME=CampusNet&PRGNAME=COURSEDETAILS&ARGUMENTS=-N749640737525148,-N000274,-N384332886947432,-N385935390323247,-N385935390316248,-N0,-N0).

## Scripts
### Electro thermal simulation
`electro_thermal.ipynb` is a simulation of an electric current problem coupled with a thermostatic simulation as a postprocessing step. An electric copper conductor with a narrow section is the setup for the simulation.
The PDEs are solved using the [Finite integation technique](https://de.wikipedia.org/wiki/Finite-Integral-Methode). The FIT code can be found in the `/fit` folder.

![](https://github.com/Devoev/multiphysics/blob/master/out/j_static.png?raw=true)
![](https://github.com/Devoev/multiphysics/blob/master/out/thermo_static.png?raw=true)

### Pendulum simulation
`pendulum.ipynb` and `double_pendulum.ipynb` are simulations of a mechanical pendulum and double pendulum respectively. The initial value problems are solved by the [implicit Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method). 
Code for IVP solvers are found under `/ode`.
The nonlinearity of the pendulum equations are dealt with by a *dynamic iteration* or *waveform relaxation*. This code is found in `/nonlinear`.

![](https://github.com/Devoev/multiphysics/blob/master/out/double_pendulum.gif?raw=true)
