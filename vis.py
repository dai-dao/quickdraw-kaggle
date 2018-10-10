import matplotlib.pyplot as plt


def draw_random_images(train_X, train_y, word_encoder):
    fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
    rand_idxs = np.random.choice(range(train_X.shape[0]), size = 9)
    
    for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
        test_arr = train_X[c_id]
        test_arr = test_arr[test_arr[:, 2] > 0, :] # only keep valid points
        lab_idx = np.cumsum(test_arr[:, 2] - 1)
        # For each stroke
        for i in np.unique(lab_idx):
            c_ax.plot(test_arr[lab_idx == i, 0], 
                    np.max(test_arr[:, 1]) - test_arr[lab_idx == i, 1], '.-')

        c_ax.axis('off')
        c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])])