from keras.callbacks import EarlyStopping, ModelCheckpoint

print('Setting up...')
import os

from utils import *
from sklearn.model_selection import train_test_split

path = 'data/2'

### IMPORT DATA
#convert log file to csv for editing,...
convert_to_csv(path)
data = importData('driving_log.csv')
# print(data)

### SPLIT FOR TRAINING AND VALIDATION
imagesPath, outputList = loadData(path, data)
print(outputList)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, outputList, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))


### DATA AUGMENTATION
### PRE-PROCESSING

model = createModel()
model.summary()
model.compile(optimizer='adam', loss='mse')
### TRAINING MODEL
imgsPerStep = 100
steps = 300
epochs = 100
batchCreated = 128
valSteps = 200

batchGenerate(xTrain, yTrain, imgsPerStep, 1)

min_delta = 0.0005
patience = 5
verbose = 1

callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=patience,
                  min_delta=min_delta),
    ModelCheckpoint(monitor='val_loss',
                    filepath='models',
                    save_best_only=True,
                    verbose=verbose)]

# history = model.fit(batchGenerate(xTrain, yTrain, imgsPerStep, 1), steps_per_epoch=steps, epochs=trainingTimes,
#           validation_data=batchGenerate(xVal, yVal, batchCreated,0), validation_steps=valSteps)
history = model.fit(batchGenerate(xTrain, yTrain, imgsPerStep, 1),
                    steps_per_epoch=steps,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=batchGenerate(xVal, yVal, batchCreated,0),
                    validation_steps=valSteps,
                    verbose=verbose,
                    workers=1
                    )


#save model
model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()