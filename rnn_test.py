import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 7.20 데이터 로드 및 확인
# 데이터를 메모리에 불러옵니다. encoding 형식으로 utf-8 을 지정해야합니다.
train_text = open('pose256GB_20.csv', 'rb').read().decode(encoding='utf-8')
# train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open('pose256GB_20.csv', 'rb').read().decode(encoding='utf-8')

# 텍스트가 총 몇 자인지 확인합니다.
print('Length of text: {} characters'.format(len(train_text)))
print('Length of text: {} characters'.format(len(test_text)))
print()

# 7.21 학습을 위한 정답 데이터(Y) 만들기
#### 수정 -> 20개의 데이터 당 하나의 y값을 주기 위해서 바꿈 [[1] [0]]
train_Y = [[row.split(',')[15]] for row in train_text.split('\r')[19::20]]
test_Y = [[row.split(',')[15]] for row in test_text.split('\r')[19::20]]

# 운동 이름을 정수로 변경
list=[]
for y_train in train_Y:
    for y in y_train:
        y=y.replace('squat','0').replace('running','1')
        list.append([int(y)])

train_Y=np.array(list)
test_Y=train_Y

print("여기까지 ok")
print(train_Y)
print(test_Y)
print(train_Y.shape, test_Y.shape)

# 7.22 train 데이터의 입력(X)에 대한 정제(Cleaning)
#### 수정 -> 줄바꿈 기호 없애고 빈 문자가 아닌 경우에만 데이터에 추가
list2=[]
for row in train_text.split('\n')[0:]:
    list=row.split(',')[0:15]
    for i in list:
        if i!='':
            list2.append(i)
train_X=np.array(list2)
# train_text_X= np.array([row.split(',')[0:15] for row in train_text.split('\n')[0:]])
# train_X= np.array([row.split(',')[0:15] for row in train_text.split('\n')[0:]])
train_X=train_X.reshape(2,20,15).astype(float)
train_text_X=train_X

print("X 확인")
print(train_X)
# # train_X 만들기


print("모델 시작")
#######################################################################
model = tf.keras.Sequential([
    #### 수정 -> Embedding을 없애고 RNN 입력 데이터 형태 input_shape=[20, 15]로 바꿈
    #tf.keras.layers.Embedding(20000, 300, input_length=25),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.SimpleRNN(units=15, activation='tanh', return_sequences=True, input_shape=[20,15]),
    tf.keras.layers.LSTM(units=10),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs=10, batch_size=128, validation_split=0.2)

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
test_text_X = train_text_X
test_X = train_X

# 테스트 정확도 측정
print("테스트 정확도 측정 시작")
model.evaluate(test_X, test_Y, verbose=1)
print("테스트 정확도 측정 종료")


prediction=model.predict(test_X)
print(prediction)