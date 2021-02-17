# Natrual Language Processing Notes

+ Supervised ML
  + Features $X$
  + Labels $Y$
  + Parameters $\theta$
  + Predicted $\hat{Y}$
  + Loss：$\hat{Y}$ vs $Y$
  + e.g. Sentiment analysis
    + text: Tweet
    + Postive 1; Negative 0;
+ Vocabulary & Feature Extraction
  + $V=[ I, am, happy, ..., hate]$
  + sparse representation $[1, 1, 1,..., 0]$
  + Positive and negative counts
    + features-freq： dictionary mapping from word to frequency
    + $X_m=[1,\sum PosFreq, \sum NegFreq]$ (not dupicate)
+ Preprocessing
  + stop words and punctuation
    + and, are, is; , . ! ""
    + Handles and URLs
  + Stemming and lowercasing
    + transform words into base stem
+ Logistic Regression:
  + sigmoid $h(x^i,\theta)=\frac{1}{1+e^{-\theta^Tx^i}}$
  + Training: find $\theta$ minimize cost function
    + Initiate $\theta$ 
    + Update paramters at gradient dicrection
  + Test
    + $X_{val}$,$Y_{val}$ 
  + Cost Function
    + $J(\theta)=-\frac{1}{m} \sum_{i=1}^m [y^i log h(x^i,\theta) + (1-y^i)log(1-h(x^{i},\theta))]$