import inline as inline
import matplotlib as matplotlib
import matplotlib
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import SVG

import tensorflowjs as tfjs
import pandas as pd

# 7.20 데이터 로드 및 확인
# 데이터를 메모리에 불러옵니다. encoding 형식으로 utf-8 을 지정해야합니다.
from pasta.augment import inline

# 학습 데이터와 훈련 데이터 split
# X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 42, shuffle = True)


train_text = open('./dataset/jsontocsv.csv', 'rb').read().decode(encoding='utf-8')
# train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open('./dataset/jsontocsv.csv', 'rb').read().decode(encoding='utf-8')

# 연속 프레임 개수
n_steps=16

# 텍스트가 총 몇 자인지 확인합니다.
print('Length of text: {} characters'.format(len(train_text)))
print('Length of text: {} characters'.format(len(test_text)))
print()

# 7.21 학습을 위한 정답 데이터(Y) 만들기
#### 수정 -> 20개의 데이터 당 하나의 y값을 주기 위해서 바꿈 [[1] [0]]
train_Y = [[row.split(',')[15]] for row in train_text.split('\r')[15::16]]
test_Y = [[row.split(',')[15]] for row in test_text.split('\r')[15::16]]
label_value = {1: 0, 33: 1, 49: 2, 81: 3, 113: 4, 145: 5, 177: 6, 185:7}
label=['1', '33', '49', '81', '113','145','177','185']

# 운동 이름을 정수로 변경
train_ylist=[]
for y_train in train_Y:
    for y in y_train:
        train_ylist.append([label_value[int(y)]])

train_Y=np.array(train_ylist)

test_ylist=[]
for y_test in test_Y:
    for y in y_test:
        test_ylist.append([label_value[int(y)]])

test_Y=np.array(test_ylist)

class_num=len(np.unique(train_Y))

print("여기까지 ok")
# print(train_Y)
# print(test_Y)
# print(train_X.shape, test_X.shape)
# print(train_Y.shape, test_Y.shape)

# 7.22 train 데이터의 입력(X)에 대한 정제(Cleaning)
#### 수정 -> 줄바꿈 기호 없애고 빈 문자가 아닌 경우에만 데이터에 추가
train_xlist=[]
train_x=train_text.split('\n')[0:]
blocks=int(len(train_x)/n_steps)

for row in train_x:
    list=row.split(',')[0:15]
    for i in list:
        if i!='':
            train_xlist.append(i)
train_X=np.array(train_xlist)
train_X=train_X.reshape(blocks,n_steps,15).astype(float)

test_xlist=[]
for row in test_text.split('\n')[0:]:
    list=row.split(',')[0:15]
    for i in list:
        if i!='':
            test_xlist.append(i)
test_X=np.array(test_xlist)
test_X=test_X.reshape(blocks,n_steps,15).astype(float)


print("X 확인")
# print(train_X)
# print(test_X)
# # train_X 만들기
# print(train_X.shape, test_X.shape)


print("모델 시작")
#######################################################################
model = tf.keras.Sequential([
    #### 수정 -> Embedding을 없애고 RNN 입력 데이터 형태 input_shape=[20, 15]로 바꿈
    #tf.keras.layers.Embedding(20000, 300, input_length=25),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.MaxPooling1D(pool_size=4),

    # tf.keras.layers.SimpleRNN(units=15, activation='tanh', return_sequences=True, input_shape=[n_steps,15]),
    tf.keras.layers.LSTM(50, input_shape = (n_steps,15), return_sequences = True),
    tf.keras.layers.LSTM(50, return_sequences = False),
    tf.keras.layers.Dense(class_num, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs=50, batch_size=128, validation_split=0.2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
print(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

##test_text_X= [row.split(',')[0:15] for row in train_text.split('\r')[0:]]
##test_X= [row.split(',')[0:15] for row in train_text.split('\n')[0:]]

#### 수정 -> train과 동일한 형태로 바꿈

# 테스트 정확도 측정
print("테스트 정확도 측정 시작")
model.evaluate(test_X, test_Y, verbose=1)
print("테스트 정확도 측정 종료")


prediction=model.predict(test_X)
print(prediction)

# plot = plot_confusion_matrix(model, # 분류 모델
#                              train_X, train_Y, # 예측 데이터와 예측값의 정답(y_true)
#                              display_labels=label, # 표에 표시할 labels
#                              cmap=plt.cm.get_cmap('Blues'), # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)
#                              normalize=None) # 'true', 'pred', 'all' 중에서 지정 가능. default=None
# plot.ax_.set_title('Confusion Matrix')

# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model_shapes.png', show_shapes=True)
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# model_json = model.to_json()
# with open("model.json", "w") as json_file :
#     json_file.write(model_json)

# model.save("my_model.h5")


confusion_matrix = sklearn.metrics.confusion_matrix(test_Y, np.argmax(prediction, axis = 1))
print(confusion_matrix)

width = 8
height =8
plt.figure(figsize=(width, height))
plt.imshow(
    confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.Blues
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(class_num)
plt.xticks(tick_marks, label, rotation=90)
plt.yticks(tick_marks, label)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

