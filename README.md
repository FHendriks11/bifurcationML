# bifurcationML

In this repository, we investigate generative AI approaches for systems with bifurcation behavior that results from unstable points. This means there are sometimes multiple possible correct predictions. We want to use generative AI/probabilistic models to predict all possible options.

Corresponding paper: Hendriks, F., Rokoš, O., Doškář, M., Geers, M. G., & Menkovski, V. (2025). Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems. arXiv preprint arXiv:2509.03340.
Link: https://arxiv.org/abs/2509.03340

## Installation
The repository contains both python scripts (.py) and Jupyter Notebooks (.ipynb). The only libraries that are needed are:
* numpy
* scipy
* pytorch
* pytorch geometric
* matplotlib

## Test systems
0) Two Delta Peaks: simplest test of flow matching: map a 1D Gaussian to a probability distribution consisting of two Dirac delta peaks at +1 and -1.

1) Heads or Tails: given an amount X to bet, output the winnings (50% chance of X, 50% chance of -X). Simplest possible form of bifurcation.

2) 3 Roads: two entities must coordinate, since they cannot clash, creating a correlated probability distribution.

3) Four Node Graph: input is a graph with the same node embedding consisting of one value for each node; output is a new value for each node, which is no longer the same, breaking the permutation symmetry.

4) Buckling beams: a classic mechanics problem where a beam under compression can buckle in multiple direction. The input are the vertical displacement of the tip and parameters of the segments (stiffness, length) and joints (rotational stiffness), and the output is a multimodal distribution over trajectories.

5) Allen-Cahn: a PDE describing phase separation. The solution evolves over time and exhibits bifurcations depending on PDE certain parameters. Flow matching is used to model the distribution over possible evolutions of the mixture over time, testing the method in high-dimensional, dynamic systems.

## Support
f.hendriks@tue.nl

## Authors and acknowledgment
Fleur Hendriks. Supervisors: Vlado Menkovski, Martin Doškář, Marc Geers, Ondřej Rokoš.

## License
To be added.

## Project status
In progress
