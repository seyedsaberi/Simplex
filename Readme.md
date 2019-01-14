A brief description of each file:

main.py : This file constructs a simulated simplex with arbitrary noise and dimension. Then generates random samples from the simplex. Then uses these samples to estimate vertices of the simplex based on our proposed algorithm. Finally, calculate the precision of our algorithm by comparing its results with the true vertices.

gmgroup.py : This file tests our proposed algorithm on a biological dataset.
S. S. Shen-Orr, R. Tibshirani, P. Khatri, D. L. Bodian, F. Staedtler, N. M. Perry, T. Hastie, M. M. Sarwal, M. M. Davis, and A. J. Butte, “Cell type–specific gene expression differences in complex tissues,” Nature methods, vol. 7, no. 4, p. 287, 2010.

error.py : This file peruses the precision of our algorithm when SNR of samples decreases.

dimension.py: This file peruses the precision of our algorithm when the dimension of the simplex increases.

cuprite.py : This file tests our proposed algorithm on a hyperspectral imaging dataset.
F. Zhu, Y. Wang, B. Fan, G. Meng, S. Xiang, and C. Pan, “Spectral unmixing via data-guided sparsity,” CoRR, vol. abs/1403.3155, 2014.


rcch.py & utility.py : These files include implementation of our algorithm.



