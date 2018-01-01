import numpy as np
import pickle
from sklearn import decomposition as d
import matplotlib.pyplot as plt

n_emotions = 5
n_examples_per_emotion = 12000

margin = 4.0

state_size = 64

#ESTABLISH CENTERS
centers = np.empty([n_emotions, state_size])
count = 0
centers[0] = 2*np.random.randn(state_size)-1
count += 1
while count < 5:
	take_new = True
	new_center = np.empty([state_size])
	while take_new:
		new_center = 4*np.random.rand(state_size)-2
		take_new = False
		for j in range(count):
			dist = np.sqrt(np.sum(np.square(new_center - centers[j])))
			if dist <= 2.5*margin:
				take_new = True
				break
	centers[count] = new_center
	count += 1

print("Found centers")
two_d_decomp = d.PCA(n_components=2)

#GENERATE EXAMPLES
all_examples = np.empty([n_emotions, n_examples_per_emotion, state_size])
n_examples = 0
example_count = np.zeros([n_emotions], dtype=np.int32)
example_num = 0
while n_examples < n_emotions * n_examples_per_emotion:
	new_example = np.random.rand(state_size)
	mod = np.sqrt(np.sum(np.square(new_example)))
	if mod <= margin:
		all_examples[example_num][example_count[example_num]] = centers[example_num] + new_example
		example_count[example_num] += 1
		if example_count[example_num] == n_examples_per_emotion:
			example_num += 1
		n_examples += 1
	if n_examples % 1000 == 999:
		print(str(n_examples))

new_2d_centers = two_d_decomp.fit_transform(all_examples.reshape([-1, state_size]))
plt.scatter(new_2d_centers[:, 0], new_2d_centers[:, 1])
plt.show()

diction = {"x": all_examples, "centers": centers}
pickle.dump(diction, open("emotion_data.p", "wb"), protocol=2)