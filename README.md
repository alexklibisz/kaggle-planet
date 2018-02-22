# kaggle-planet

Code for the [Kaggle Understanding the Planet from Space Competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).

Overview

- This was a team effort with [Grant Bruer](https://github.com/gbruer15).
- The final solution ranked 103rd of ~970 teams (top 11 percent).
- The final solution was a small ensemble of ResNets and DenseNets using test-time augmentation and some trickery to weight the ensemble votes for each image. This was not terribly different from the winning solution, though the winner did use a neat de-hazing trick which was described on the Kaggle discussions. In the end there was not much gain in using an ensemble. IIRC, you could get 91% accuracy with a simple 8-10 layer conv-net. This is referred to as "Goddard" in our repository.
- The repository is fairly messy; all of the models are convolutional networks implemented in Keras.
- Grant and I made a [Google Doc outlining some of the things we learned](https://docs.google.com/document/d/16ndg9J4gsjFAjZ6MINuSGmz2Bw0e51IGA1FB930LX9o/edit#heading=h.ivbqik308omd)
