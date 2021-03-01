# Torch for R practice
Place to practice using the R machine learning framework Torch.

1. [Guess the correlation](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-cpu.md)</br>
Adapted from https://torch.mlverse.org/start/guess_the_correlation/
  * [Alternate version](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-cpu-no-enumerate.md) that doesn't use torch "enumerate" function. As of 2/22/2021, the torch "enumerate" function doesn't work with GPUs.

2. What if? Experiments and adaptations</br>
I used a GPU for all of these examples.
  * [What if … we wanted to train on GPU?](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-gpu.md)</br>
Adapted from https://torch.mlverse.org/start/what_if/#what-if-we-wanted-to-train-on-gpu
  * [What if … we wanted to work with linear data?](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-gpu-linear-input.md)</br>
Adapted from https://torch.mlverse.org/start/what_if/#what-if-we-were-working-with-a-different-kind-of-data-not-images
  * [What if … we wanted to classify?](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-gpu-classify.md)</br>
Adapted from https://torch.mlverse.org/start/what_if/#what-if-we-wanted-to-classify-the-images-not-predict-a-continuous-target
  * [What if … we wanted to add regularization?](https://github.com/iparakati/torch-for-R-practice/blob/main/guess-the-correlation-gpu-bigger.md)</br>
Adapted from https://torch.mlverse.org/start/what_if/#what-if-we-made-the-network-bigger-or-trained-it-for-a-longer-time