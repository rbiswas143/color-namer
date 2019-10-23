## Naming colors with Machine Learning

This work is a re-creation of [Micheal A. Alcorn's](https://github.com/airalcorn2) [work](https://github.com/airalcorn2/Color-Names)
on Github. Also checkout his interesting [blog post](https://opensource.com/article/17/9/color-naming-word-embeddings)
about the same

### Objectives
- Given the name of a color, predict the color (RGB vlaue)
- Given a color as an RGB value, suggest some new names for it

Neural networks can learn to assign colors to color names and predict new names for colors. The primary challenge that we face in
this process is that the largest color databases are rather small and most of the words in the vocabulary of the language do not
appear in any color name. As such we end up with models with poor generalization to the language. This can, to some extent, be
handled by the use of transfer learning. We represent color names as word embeddings derived from a pre-trained word embeddings
dataset. The neural networks now learn a mapping between the word embedding and color spaces, which greatly improves the
generalization of the models even for out-of-vocabulary words

### Datasets
- *Small* - 1000+ colors from Sherwin Williams's
[collection](https://images.sherwin-williams.com/content_images/sw-colors-name-csp-acb.acb) of colors
- *Big* - [Dictionary](https://github.com/meodai/color-names) of 18,000+ handpicked colors from various sources
- *Word Embeddings* - 6B word ebeddings from [GloVe](http://nlp.stanford.edu/data/glove.6B.zip)

### Deep Learning Architectures
- **Color Predictions**
  - *RNN/LSTM based sequence models*: These models sequenctially process the words in a color name and predict an RGB value
at the last time step
  - *CNN model*: The input is truncated and padded to a fixed length ehich is then reduced by a CNN model to an RGB value
- **Color Name Predictions**
  - *RNN/LSTM based sequence models*: These are trained in the style of language modelling where the input to the first time
  step is a resized color RGB value vector, while the subsequent time steps take as input the output of the previous time
  step. At each time step a word of the color name is predicted terminating at a fixed length or, optinally, a stop word.
  During evaluation, we can sample out multiple name predictions using Beam Search
  
### Project Structure


### Results
