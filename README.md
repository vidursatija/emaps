# Emotional Maps
## Abstract
Extensive research has already been going on in neuroscience and psychology to understand the impact and effect of music, rather any given stimuli, on our mood and emotions. Researchers have concluded many facts about which factors affect our emotions. On the chemistry side, we know which neurotransmitters are majorly responsible for changing our mood(like dopamine, norepinephrine and serotonin. We also know how these are produced and to which portions of the brain they travel to. This project is related to the mathematical part of emotions, and using deep learning I've found quite impressive results on how our brain is affected by any given stimuli. 

## Experimenting with your own stimulus
1. Take your set of stimulus and apply PCA. 64 components are used my example but you're free to choose any. Find the projection weights. Apply KMeans with k=5(preferrably as there are 5 major emotions). 

2. Use the following code
```python
STE = StimulusToEmotion("mnist_projection_weights.p", tf.constant(features), sess)
DE = DynamicEmotions(0.01, m_pointer=STE.projected_stimulus, location_tvars=STE.get_trainable_vars())
sess.run(tf.global_variables_initializer())
initial_m = sess.run(DE.m)
ee = np.reshape(centers[0], [1, -1]) #Initial emotion = first center.
for i in range(n_steps):
          reward, distance, emotion_change, ee, _, new_m = sess.run([DE.r, DE.distance, DE.emotion_change, DE.e_plus, DE.location_train_op, DE.m], feed_dict={DE.e: ee})
          delta_m = np.sum(np.square(new_m - initial_m), -1)
          initial_m = new_m
          print(" ".join(["Old Reward", str(np.squeeze(reward)), "New Distance", str(np.squeeze(distance)), "\nChange in E", str(np.squeeze(emotion_change)), "Change in M", str(np.squeeze(delta_m))]))
          print("-----------")
```
  * *features* is your given stimulus.
  * *StimulusToEmotion* and *DynamicEmotions* are classes present in model.py
  * *centers* is 2d numpy matrix which stores the centers.

3. Run the code to see how the distances between the old emotion and new emotion, and the distance between emotion and the mapped stimulus changes

### MNIST Example
In my above project, I have used the MNIST dataset and a pretrained MNIST classifier. I have used the fc4 layer of the model. 
Then I have projected it to my 64 coordinate map, and found the projection vectors and centers.
run_stimulus.py contains the above code mentioned. 
