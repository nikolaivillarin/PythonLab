from Chapt3.multilabelClassification import *

noise = np.random.randint(0, 100, (len(X_train), 784))

X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))

X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

# Clean the image and preview
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[0]])

plot_digits(clean_digit)