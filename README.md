# Musical Instrument Identification From a Soundtrack

In both modern and classical music, very often there is a wide range of musical instruments playing in
unison. This project attempts to identify what instruments are playing in a given soundtrack using a deep neural network. 

## Network Architecture
This project uses pre-trained `resnet18` without the last layer, concatenated with a dedicated fully-connected layer. Since we attempt to identify multiple instruments at once, at the network's end there is a sigmoid layer implying that every element of the output vector represents the probability that the matching instrument is present in the soundtrack.
In total, we identify 11 different instruments. 

For the model's implementation and training process, please see `dedicated_layers_model.py`.

## The MusicNet Dataset
The network was trained using [MusicNet Dataset](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset).
The dataset contains 330 classical recordings and maps which instrument is playing which note at any given time. 

### Re-indexing the dataset
This dataset has no official pytorch implementation, and we've created one under `MusicNetManyhotNotes.py`. The labeling data provided by this dataset contains an entry for every note, meaning it does not support identifying multiple notes and instruments at once. 
We re-index the dataset metadata by pre-processing the metadata provided. We combine multiple entries that refer to the same timespan thus referring to a single sample as all notes and instruments played on a certain timespan

### Caching Mechanism 
The dataset weighs `~35GB`, and loading it at once was not feasible on our hardware. Moreover, different samples are located in different `.wav` files. This affects the training process compute time, since loading two samples that originated in different `.wav` files including accessing the disk, loading the `wav` file and cropping it to the correct timespan.  
To address this problem, we perform a "block-shuffle" and caching. The metadata is sorted so that several sequential lines all originate form the same wav (while being sampled from different places in time). Our dataset implementation compliments this sorting, and will cache the `wav` loaded until a different `wav` is required. 
This leads for a single `wav` load for every block in the sorted metadata file, reducing the amount of disk access operations by a large factor, while sacrificing to a certain degree the stochastic nature of the SGD process.

## performance analysis
We compared the model performance when trained using different loss functions, and when using different methods to calculate the prediction based on the network's output.

### Different loss functions
Our dataset includes 11 instruments, with at most 3 instruments playing at the same time, and the
MSE loss counts how many predictions we got wrong. So, assuming that nothing is playing will be correct
for 8/11 instruments at minimum. While this is technically correct, it is not useful at all to us.
We understood we needed something to mitigate the effects of the false negatives that the MSE loss
caused, so we took inspiration from computer vision, specifically image segmentation, and used something
similar to the IoU loss.

We used the following loss function:

$$\scriptL = \frac{FN}{N_{playing}} + \frac{FP}{N_{not playing}}$$

This proposal gives the false positives and the false negatives the same weight, forcing the model to avoid prediction with high amount of false negatives. Using this loss function resulted in a drastic improvement in performance as can be seen under `Performence analysis/compare_loss_function`, and under the attached PDF.

### Different predictors
As shown under `Performence analysis/compare_different_predictors` and the attached PDF, the `MusicNet` dataset instrument distribution is not uniform, and some instruments are more common in the dataset then others.
We addressed this bias with a careful choise of the predictor function. 

We compare the performance of the choosing the playing instrument based on class-specific threshold, based on the dataset's instrument distribution statistic, its prediction statistics on the validation set and its prediction statistics on random noise. Also, we compare these predictors with the top-n highest elements in the output vector.

We found the the best predictor is the top-3 predictor.

## Testset results
Using the top-3 predictor, and a model trained using the IoU-like loss function, we got an *accuracy of 76%*, meaning the model correctly identified 76% of all instruments
playing. On the other hand we received, on average, *1.6 false positives*, meaning for every time frame
the model added 1.6 instruments which weren’t playing.

Ultimately, we are happy with the results. As mentioned, previous works in the subject identified
the instrument playing with an accuracy of 80%-85%. But, as this was only one instrument, we do not
think that our results are worse in any way. The false positives aren’t great, but are an obvious side
effect of choosing the top 3 instruments no matter what. Perhaps a mix between top-n and class-specific
thresholding can give better results.

References
[1] https://github.com/micmarty/Instronizer
[2] https://github.com/biboamy/instrument-prediction
