# dissonance-data-gen

## Intro
This repository investigates both data generation and the capacity of deep learning to predict chords in musical compositions using musical features extracted from midi files and synthesized audio. It encompasses multiple stages of data preprocessing and manipulation, including drum validation and removal from midi files, conversion of midi to audio, audio segmentation into windows, Constant-Q Transforms (CQT) generation, and chord synthesis from audio signals.

Following data preprocessing, a Convolutional Neural Network (CNN) is trained and tested on the data to assess the model's performance in predicting chords from audio. The results indicate that the model accurately predicts roughly 39.6% of the chords in the test set, which is a relatively low accuracy score due to the simplicity of the model, and it can be enhanced by adjusting the hyperparameters or adopting a more intricate model architecture.

As part of future research, dissonance values are to be factored in as a slider that can control the predicted chords, perhaps either as a loss function or a model conditioner. Nevertheless, the result is a significant stride towards comprehending the potential of deep learning in predicting chords in musical compositions, and it lays the groundwork for further research in this domain.
