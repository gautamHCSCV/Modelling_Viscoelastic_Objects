# Modelling_Viscoelastic_Objects
This paper proposes an alternative data-driven hap-
tic modeling method of homogeneous deformable objects based
on a CatBoost approach – a variant of gradient boosting machine
learning approach. In this approach, decision trees are trained
sequentially to learn the required mapping function for modeling
the objects. The model is trained on the input feature vectors
consisting of position, velocity and filtered velocity samples to
estimate the response force. Our approach is validated with
a publicly available two-finger grasping dataset. The proposed
approach can model unknown interactions with good accuracy
(relative root mean squared error, absolute relative error and
maximum error less than 0.06, 0.18 and 0.76 N, respectively)
when trained on just 20% of the training data. The CatBoost-
based method outperforms the existing data-driven methods both
in terms of the prediction accuracy and the modeling time when
trained on similar size of the training data.

![image](https://github.com/gautamHCSCV/Modelling_Viscoelastic_Objects/assets/65457437/866148a2-595a-4fbb-9589-84c4183c63a4)



Paper link: https://ieeexplore.ieee.org/document/10224477

