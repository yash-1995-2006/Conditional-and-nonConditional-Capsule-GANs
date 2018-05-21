# Conditional-and-nonConditional-Capsule-GANs
This is the repository for the code used in writing the paper, 'Generative Adversarial Network Architectures For Image Synthesis Using Capsule Network'.
This paper proposes GAN architectures for incorporating Capsule networks for conditional and non-conditional image synthesis. The paper also demonstartes that such an architecture requires significantly lesser training data to generate good quality images in comparison to current architectures for image synthesis (DCGANs with Improved Wasserstein Loss). The paper also demonstrates an increase in the  diversity of images generated owing to the robustness of Capsule GANs to small affine transformations.
## Architectures
### Discriminative Capsule GAN
The following diagram shows the architecture for the Discriminative Capsule GAN that is used for non-conditional image synthesis.

![CapsGANArch](Images/CapsGAN.png?raw=true "Discriminative Capsule GAN Architecture")<!-- .element height="50%" width="50%" -->

The discriminator has been substituted with a Capsule Network in place of a CNN. Also, the loss uses marginal losses described in the paper, "Dynamic Routing Between Capsules" by Sabour et al[1], for Capsule Networks to build a function analogous to the Wasserstein Loss, allowing the architecture to benefit from stable training and faster convergence of critic to optimality. 

### Split Auxiliary Conditional Capsule GAN
The following diagram shows the architecture for the Split-Auxiliary Conditional Capsule GAN that is used for conditional image synthesis.
![Alt text](Images/conditionalGAN.png?raw=true "Split-Auxiliary Conditional Capsule GAN Architecture")
