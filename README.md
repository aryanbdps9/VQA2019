<!---
I'm providing a list of headers that the README MUST have below. You're free to add more content if you wish:
1. A brief abstract of your project including the problem statement and solution approach. If the project has cool visual results, you may provide one image or GIF of the results. See this for reference.
-->
# VQA : A CS763 Odyssey
## ABSTRACT
### Problem Statement
Given an input image and a question that can be answered by looking at the image.  
### Our Approach
Building on the idea of [this](https://arxiv.org/pdf/1708.02711.pdf) paper and [this](https://github.com/hengyuan-hu/bottom-up-attention-vqa) code, Our main expansion was following:
* Change the word embeddings used in the project to ELMO as the latter uses the semantics over a sentence and hence is a boost from earlier GloVe which is a simple word to vector mapping.
* Add an extra Square Distance loss in the training loop to output word embedding of answer which is then decoded back to word by another sister neural network which is a LSTM network taking word vector as input and characterwise predicts the output word corresponding to the embedding vector input.
* Add a side classifier which takes input the embedded form of question and outputs whether its a YES/NO type answerable question and if that is true, the whole search space of answer collapses to two and predicted word embedding should be near to true vector and farther from the false vector answer.

<!---
2. A list of code dependencies.
-->
## Code Dependencies

Please use Python2 for running this code
#### Packages required
* Cpickle
* torch
* numpy
* tensorboardX
* h5py

<!---
3. Detailed instructions for running the code, preferably, command instructions that may reproduce the declared results. If your code requires a model that can't be provided on GitHub, store it somewhere else and provide a download link.
-->
## Setup


<!---
4. Results: If numerical, mention them in tabular format. If visual, display. If you've done a great project, this is the area to show it!
-->
## Results
<!---
5. Additional details, discussions, etc.
-->
## Additions Details
### Approaches Tried

6. References.
## Reference
1. [Tips and Tricks for Visual Question Answering:Learnings from the 2017 Challenge](https://arxiv.org/pdf/1708.02711.pdf)  
2. [Their implementation of the above paper](https://github.com/hengyuan-hu/bottom-up-attention-vqa)
