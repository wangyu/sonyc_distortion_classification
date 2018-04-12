# sonyc_distortion_classification
Build generalized distortion classifier for SONYC audio data using active learning

## Active Learning readings and tutorials:
**scikit-learn Semi-supervised**  
http://scikit-learn.org/stable/modules/label_propagation.html
https://en.wikipedia.org/wiki/Semi-supervised_learning#See_also 

**modAL introduction**
https://cosmic-cortex.github.io/modAL/#introduction

**Psedu Code**
```
while (convergence_rate < 0.80)
    sensorId, timestamp = get_datapoint_to_be_labelled
    wait:
        play_audio(sensorId, timestamp)
        input label ~ {1 | 0}
    add labeled datapoint to seed
    remove labeled datapoint from pool
    retrain
    evaluate - print accuracy numbers
```

## Reading:
### Web Resources:
Datacamp General Intro: https://www.datacamp.com/community/tutorials/active-learning


### Papers:


Virginia R., Learning Classification with Unlabeled Data, https://papers.nips.cc/paper/831-learning-classification-with-unlabeled-data.pdf 

Burr S., Active Learning Literature Survey http://burrsettles.com/pub/settles.activelearning.pdf

Ozan S., 2017, Active Learning for CNN: A Core-Set Approach, https://arxiv.org/abs/1708.00489v2

Yanyao S., 2017, Deep Active Learning for Named Entity Recognition, https://arxiv.org/pdf/1707.05928v2.pdf

Zhu and Bento, 2017, Generative Adversarial Active Learning, https://arxiv.org/abs/1702.07956v5

Fang et. al, 2017, Learning how to active learn: A Deep Reinforcement Learning Approach, https://arxiv.org/abs/1708.02383v1

Fang et. al, 2017, A Meta-Learning Approach to One-Step Active Learning, https://arxiv.org/abs/1706.08334



