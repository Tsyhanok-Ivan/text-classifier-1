import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

from pprint import pprint

tf.compat.v1.disable_eager_execution()

print("TensorFlow version:", tf.__version__)

# створюємо масив з категоріями
categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
# створюємо дві змінні з даними
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# виводимо к-ть текстів у даних (це масив)
print('total texts in train: ', len(newsgroups_train.data))
print('total texts in test: ', len(newsgroups_test.data))

pprint(list(newsgroups_train.target_names))

# виводимо елемент з масиву даних (текст)
print('text: ', newsgroups_train.data[1200])
# виводимо категорію цього ж тексту
print('category: ', newsgroups_train.target[1200])

# створюємо змінну для перелічення слів
vocab = Counter()
# перебираємо елементи масиву даних (тексти)
for text in newsgroups_train.data:
    # в кожному тексті перебираємо слова і збільшуємо довжину vocab
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

# виводимо загальну к-ть слів в масиві даних
# print('words total in data: ', len(vocab)) #119930

# створюємо змінну, яка буде містити довжину масива vocab
total_words = len(vocab)


# функція, яка шукає індекси слів
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i
    return word2index


# окрема змінна, в яку записано результат роботи функції
word2index = get_word_2_index(vocab)


# виводимо індекс певного слова, звертаючись до результату
# print("Index of word 'age': ", word2index['age'])

# функція перетворення тексту на "векторну модель"
def text_to_vector(text):
    layer = np.zeros(total_words, dtype=float)  # масив з нулів типу float такоъ ж структури, як total_words
    for word in text.split(' '):  # дня кожного слова в новому масиві додаємо 1
        layer[word2index[word.lower()]] += 1
    return layer


# функція, яка перетворює категорію на "векторну модель"
def category_to_vector(category):
    y = np.zeros((3), dtype=float)  # створюємо масив з трьох нулів
    if category == 0:  # якщо категорія "0", то векторне значення - 100
        y[0] = 1
    elif category == 1:  # якщо категорія "1", то векторне значення - 010
        y[1] = 1
    else:  # якщо категорія не "1" і не "0" (а "2", наприклад), то векторне значення - 001
        y[2] = 1

    return y


# функція векторизації партій текстів
# df - набір даних, i - з якого текста, batch_size - по який працюємо
def get_batch(df, i, batch_size):
    batches = []
    results = []
    # поміщаємо в texts всі тексти з df.data від і до batch_size
    texts = df.data[i * batch_size: i * batch_size + batch_size]
    # поміщаємо в categories всі категорії з df.data від і до batch_size
    categories = df.target[i * batch_size: i * batch_size + batch_size]

    # векторизуємо кожен текст (наша ф-я text_to_vector())
    for text in texts:
        layer = text_to_vector(text)
        batches.append(layer)

    # векторизуємо кожну категорію (наша ф-я category_to_vector())
    for category in categories:
        y = category_to_vector(category)
        results.append(y)

    # повертаємо отримані векторизовані тексти і катгорії до них
    return np.array(batches), np.array(results)


# виводимо результати
print("К-ть текстів та слів у 1-100 текстах: ", get_batch(newsgroups_train, 1, 100)[0].shape)
print("К-ть текстів та категорій у 1-100 текстах: ", get_batch(newsgroups_train, 1, 100)[1].shape)
# print("Векторизовані дані: ", get_batch(newsgroups_train, 1, 100))

# змінні-параметри навчання моделі
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# мережеві параметри
n_hidden_1 = 100  # 1 шар (слой) для обробки
n_hidden_2 = 100  # 2 шар (слой) для обробки
n_input = total_words
n_classes = 3  # кількість категорій для класифікації

input_tensor = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="input")
output_tensor = tf.compat.v1.placeholder(tf.float32, [None, n_classes], name="output")


# Функція обчислення кінцевих даних (при тренування моделі)
def multiplayer_perception(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Прихований слой
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Вихідний шар
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}

# модель
prediction = multiplayer_perception(input_tensor, weights, biases)

# змінна з вимірами можливості помилки
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))

# змінна зі значеннями оптимізації (мінімізація помилок при класифікації даних при навчанні)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# ініціалізація змінних (variables)
init = tf.compat.v1.global_variables_initializer()

# Зберігання змінних (для можливості відновлення)
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    # Тренувальний цикл
    for epoch in range(training_epochs):
        avg_cost = 0.  # float type
        total_batch = int(len(newsgroups_train.data) / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
            c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))
    print("Optimization finished!")

    # тест
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in: %s" % save_path)

print("--------------------------------")

x_10_texts, y_10_correct_labels = get_batch(newsgroups_test, 0, 10)

rc = []
print("REAL CATEGORIES: ")
for i in range(10):
    rc.append(newsgroups_test.target[i])
print(rc)

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored")

    classification = sess.run(tf.argmax(prediction, 1), feed_dict={input_tensor: x_10_texts})
    print("Predicted categories: ", classification)


print('Робота програми завершилася. Помилок немає')
