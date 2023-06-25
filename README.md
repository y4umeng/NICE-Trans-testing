# NICE-Trans: Non-iterative Coarse-to-fine Transformer Networks for Joint Affine and Deformable Image Registration
Image registration is a fundamental requirement for medical image analysis. Deep registration methods based on deep learning have been widely recog-nized for their capabilities to perform fast end-to-end registration. Many deep registration methods achieved state-of-the-art performance by perform-ing coarse-to-fine registration, where multiple registration steps were iterat-ed with cascaded networks. Recently, Non-Iterative Coarse-to-finE (NICE) registration methods have been proposed to perform coarse-to-fine registra-tion in a single network and showed advantages in both registration accura-cy and runtime. However, existing NICE registration methods mainly focus on deformable registration, while affine registration, a common prerequisite, is still reliant on time-consuming traditional optimization-based methods or extra affine registration networks. In addition, existing NICE registration methods are limited by the intrinsic locality of convolution operations. Transformers may address this limitation for their capabilities to capture long-range dependency, but the benefits of using transformers for NICE reg-istration have not been explored. In this study, we propose a Non-Iterative Coarse-to-finE Transformer network (NICE-Trans) for image registration. Our NICE-Trans is the first deep registration method that (i) performs joint affine and deformable coarse-to-fine registration within a single network, and (ii) embeds transformers into a NICE registration framework to model long-range relevance between images. Extensive experiments with seven public datasets show that our NICE-Trans outperforms state-of-the-art reg-istration methods on both registration accuracy and runtime.  

**Upcoming! - Official code for MICCAI2023 paper entitled "Non-iterative Coarse-to-fine Transformer Networks for Joint Affine and Deformable Image Registration".**  
