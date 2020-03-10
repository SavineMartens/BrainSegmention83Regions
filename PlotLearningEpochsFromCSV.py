import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_history.csv')

print(df)

epochs = range(1, 199 + 2)
acc = df['dice_coef_prob']
val_acc = df['val_dice_coef_prob']

loss = df['loss']
val_loss = df['val_loss']

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(epochs, acc, 'bo', color='red', label='Training')
plt.plot(epochs, val_acc, 'b', color='blue', label='Validation')
plt.title('Dice coefficient (probability) during training' )
plt.legend()
plt.ylim(0, 1)

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, 'bo', color='red', label='Training')
plt.plot(epochs, val_loss, 'b', color='blue', label='Validation')
plt.title('Dice loss' )
plt.legend()
plt.xlabel('Epochs')
# plt.ylim(0, 1)

plt.show()