<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});
</script>
<script type="text/javascript"
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Machine Learning Engineer Nanodegree

## Capstone Proposal
Xin Chen  
August 21st, 2017

## Proposal

### Domain Background

Understanding the interaction model and stability of matter at the atomic level is of importance in many disciplines ranging from molecular synthesis to design of new materials. A synthesis chemist would – whether intendedly or unintendedly – combine basic chemistry textbook concepts, such as Lewis formulas, valence bond theory, and aromaticity counts with life-long experience and be able to name weak groups in organic compounds. Likewise, an inorganic chemist may discuss the stability of local defects in crystals based on crystal field theory, bond-order models, and the like.  The local stability information subsequently allows for rational design of new reaction routes and invention of new materials. Such concept-driven analytic chemistry has over the past decades been supported by elaborate first-principles calculations using density functional theory (DFT), and highly reliable DFT calculations may now be done routinely resulting in the buildup of computed structure-energy databases. This has opened a new route to identifying local stability in matter: the machine learning route.
In the machine learning (ML) approach[1-12], computational efforts are spent based on existing structure-energy relations for entire atomic structures (whether molecules, clusters, or solids) to build models that may predict total energies of unknown structures. Depending on the ML model, the total energy may be expressed as a sum over local energies. Inspired by the early work of Behler and Parrinello[2], a number of implementations using artificial neural networks have emerged, where the sum is over atomic energies[1, 2, 4, 5, 12]. All of these works proved the mathematical usefulness of atomic energies (or similar concepts). However, most of previous machine learning research focused on interpreting datasets composed of only chemically reasonable structures, which may significantly limit the application potential. Furthermore, very few of them made an attempt to interpret and validate the concept of atomic energy, for instance by incorporating atomic energies into chemical theory or utilizing them in research-level applications.

### Problem Statement

The goal of this project is to develop a new machine learning framework for understanding quantum chemistry at the atomic level. The following requirements must be fulfilled:
1. The features should uphold all three spatial invariances
	* Translational invariance
	* Rotational invariance 
	* Permutational invariance
2. The model should be able to accurately model the total energies given structural data.
3. The model should be able to learn local properties (atomic energies).
4. The derived atomic energies should agree with chemistry theories, e.g. the valence-bond theory.
5. The model should be interpretable.

### Datasets and Inputs

For this project we have three datasets to study:

1. QM7
	- A public dataset. 
	- This is a subset of the huge database GDB-13, and has 7165 different stable organic molecules of up to 23 atoms (C, H, N, O, S).
	- URL: [http://quantum-machine.org/datasets/#qm7][1]
2. GDB-9
	- A public dataset.
	- This dataset contains 133,885 stable organic molecules of up to 29 atoms (C, H, N, O, F).
	- URL: [https://www.nature.com/articles/sdata201422][2]
3. $\mathrm{C}_9\mathrm{H}_7\mathrm{N}$
	- DFTB: this dataset is obtained randomly generated using the evolutionary algorithm[14] implemented in ASE[15]. All structures are optimized with the density functional tight binding (DFTB) theory[16] using DFTB+ [17] in a `25x25x10` ￼Å unit cell including only forces in the x- and y- directions to produce two-dimensional structures . The bond parameters are from Gaus et al[18]. 
	- PBE: this dataset is obtained by re-optimizing the DFTB dataset with GPAW[19] using PBE in LCAO mode and DZP basis set. 

QM7 and GDB-9 are some of the most common datasets to test the performance of machine learning models in chemistry. However, these two datasets only contain chemically reasonable structures and even the linear model can give acceptable performances (\< 1.0 eV) on these datasets. So here we also use the $\mathrm{C}_9\mathrm{H}_7\mathrm{N}$ (both PBE and DFTB) datasets that consists of chemically disordered (scrap) data.

![][image-1]

### Solution Statement

To accomplish all goals listed in **Problem Statement**, a many-body expansion theory (MBE) based 1D-convolutional neural network model may be the best choice. The MBE scheme has proved successful in many theoretical chemical studies and for 1D convolutional neural networks their property of supporting variable-length inputs allows us to construct a neural network capable of processing atomistic structures of varying size and composition. Alexandrova et al[9] already demonstrated the strength of CNN in chemistry. However, their MBE-NN-M model is not permutational invariant and cannot predict atomic energies. 

### Benchmark Model

In this project we have three benchmark models:

1. MBE-NN-M
	- Proposed by Alexandrova et al[9] in 2016
	- Permutational invariance is not kept
	- MAE of 0.2 eV on their own datasets: $\mathrm{Pt}_{9}$ and  $\mathrm{Pt}_{13}$
	- The model is based on convolutional neural network
	- Not able to provide atomic energy
2. DTNN
	- Deep tensor neural network proposed by Schutt et al[5] in 2017 
	- Uphold all three spatial invariance
	- 1.0 kcal/mol (0.05 eV) on GDB-9
	- Has the concept of atomic energy 
	- No theoretical proof of their atomic energy
3. Coulomb Matrix based multiplayer perceptron
	- Proposed by Rupp et al[7] in 2012
	- Not invariant to permutation
	- Not able to provide atomic energy
	- 3.5 kcal/mol (0.18 eV) on QM-7

All benchmark models mentioned above haven’t discussed about how neural networks work in their research.

### Evaluation Metrics

1. Spatial invariance:
	- Proof of the translational invariance must be given
	- Proof of the rotational invariance must be given
	- Proof of the permutational invariance must be given
2. Total energy:
	- Metric: mean absolute error (MAE)
		- $\mathrm{MAE} = \frac{1}{n}\sum_{i=i}^{n}{| y_{i} - \hat{y}_{i} |}$
	- The MAE should be less than the DFT accuracy (0.2 eV or 5 kcal/mol)
3. Atomic energy:
	- Proof of the atomic energies based on chemistry theory must be provided. 
4. Understanding the learning:
	- Explanation of how the model really works should be given.

### Project Design

To complete this project a possible workflow could be organized as follows:

1. Theoretical analysis of the many-body expansion theory to determine which terms should be included:
$$ E^{total} = \sum_{a}^{C^N_1}{E^{(k=1)}_{a}} + \sum_{a,b}^{C^N_2}{E^{(k=2)}_{ab}} + \sum_{a,b,c}^{C^N_3}{E^{(k=3)}_{abc}} + \sum_{a,b,c,d}^{C^N_4}{E^{(k=4)}_{abcd}} + \cdots $$
2. Design the input features with all spatial invariances upheld. 
3. Design the structure of the CNN.
4. Implement the CNN model based on Google’s TensorFlow.
5. Validate the performance of the model on predicting total energies using datasets mentioned above.
6. Validate the atomic energy obtained on the $\mathrm{C}_9\mathrm{H}_7\mathrm{N}$ (PBE) dataset.
7. Try to visualize the CNN to make clear how it works in learning chemistry.

### References

1. Behler, J., Neural network potential-energy surfaces in chemistry: a tool for large-scale simulations. Phys Chem Chem Phys, 2011. 13: p. 17930-17955.
2. Behler, J. and M. Parrinello, Generalized neural-network representation of high-dimensional potential-energy surfaces. Phys Rev Lett, 2007. 98(14): p. 146401.
3. Yao, K., J.E. Herr, and J. Parkhill, The many-body expansion combined with neural networks. J Chem Phys, 2017. 146(1): p. 014106.
4. Yao, K., et al., Intrinsic Bond Energies from a Bonds-in-Molecules Neural Network. The Journal of Physical Chemistry Letters, 2017. 8(12): p. 2689-2694.
5. Schutt, K.T., et al., Quantum-chemical insights from deep tensor neural networks. Nat Commun, 2017. 8: p. 13890.
6. Hansen, K., et al., Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies. J Chem Theory Comput, 2013. 9(8): p. 3404-19.
7. Rupp, M., et al., Fast and accurate modeling of molecular atomization energies with machine learning. Phys Rev Lett, 2012. 108(5): p. 058301.
8. Hansen, K., et al., Machine Learning Predictions of Molecular Properties: Accurate Many-Body Potentials and Nonlocality in Chemical Space. J Phys Chem Lett, 2015. 6(12): p. 2326-31.
9. Zhai, H. and A.N. Alexandrova, Ensemble-Average Representation of Pt Clusters in Conditions of Catalysis Accessed through GPU Accelerated Deep Neural Network Fitting Global Optimization. Journal of Chemical Theory and Computation, 2016: p. acs.jctc.6b00994-14.
10. Dolgirev, P.E., I.A. Kruglov, and A.R. Oganov, Machine learning scheme for fast extraction of chemically interpretable interatomic potentials. AIP Advances, 2016. 6(085318).
11. Peterson, A.A., Acceleration of saddle-point searches with machine learning. The Journal of chemical physics, 2016. 145.
12. Khorshidi, A. and A.A. Peterson, Amp: A modular approach to machine learning in atomistic simulations. Computer Physics Communications, 2016. 207: p. 310-324. 
13. Oganov, A.R. and M. Valle, How to quantify energy landscapes of solids. J Chem Phys, 2009. 130(10): p. 104504.
14. Jorgensen, M.S., M.N. Groves, and B. Hammer, Combining evolutionary algorithms with clustering toward rational global structure optimization at the atomic scale. J Chem Theory Comput, 2017.
15. Larsen, A.H., et al., The atomic simulation environment-a Python library for working with atoms. Journal of Physics-Condensed Matter, 2017. 29(27). 
16. Gaus, M., A. Goez, and M. Elstner, Parametrization and Benchmark of DFTB3 for Organic Molecules. Journal of Chemical Theory and Computation, 2013. 9(1): p. 338-354. 
17. [https://wiki.fysik.dtu.dk/gpaw/index.html][3]

[1]:	http://quantum-machine.org/datasets/#qm7
[2]:	https://www.nature.com/articles/sdata201422
[3]:	https://wiki.fysik.dtu.dk/gpaw/index.html

[image-1]:	./images/figureS2.png