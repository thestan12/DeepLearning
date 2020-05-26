from src.main import *

import src.display as display

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()
    model = create_model()

    logs = model.fit(x_train, y_train, epochs=EPOCH,
                     batch_size=BATCH_SIZE,
                     validation_data=(x_test, y_test), verbose=2)

    # Sauvegarde des résultats dans un fichier excel
    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plotAccName = display.getRamdomFileName()

    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.savefig("image/" + plotAccName)

    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plotAccValName = display.getRamdomFileName()

    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.savefig("image/" + plotAccValName)

    # show_confusion_matrix(model, x_test, y_test, show_errors=True)
    img = [display.display_img(x_train[0]), display.display_img(x_train[1]), plotAccName, plotAccValName]
    displayTab = [DATASET_PATH, str(x_train.shape[0]), BATCH_SIZE, EPOCH]

    display.addRow(displayTab, img)

    # Sauvegarde du model
    model.save("my_model.keras")