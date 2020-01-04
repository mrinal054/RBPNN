## Radial Basis Probabilistic Neural Network (RBPNN)

This code performs Subtractive clustering-based 
radial basis probabilistic neural network (RBPNN).
It is a combination of probabilistic neural network 
(PNN) and radial basis function network (RBFN).
This code can be used for any classification problem.

User will need to provide following input parameters - 
- `s_input` --> row-wise training set. Example: 700x64 

(Let's say we have 700 images of 100 different persons. So, we have 7 images per person. 
After applying a feature extraction technique on 700 images, let's say each will have 64 features. 
Features can be extracted using PCA, DCT, ICA etc.
So, `s_input` is basically the training set after feature extraction.)
- `s_test` --> row-wise testing set. Example: 700x64
- `TA` --> no. of targets
- `PA` --> no. of images per train person;
- `PAT` --> no. of images per test person;
- `L` --> learning rate

The code will generate two outputs - 
- `out1` --> returns list of matched persons
- `out2` --> returns recognition rate

#### Reference
Mrinal Kanti Dhar and Md. Sanwar Hussain, Robust Face Recognition Using Radial Basis Probabilistic Neural Network, International Conference on Electrical & Electronic Engineering (ICEEE-2017).