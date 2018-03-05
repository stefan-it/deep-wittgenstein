from magpie import Magpie

with open('categories.labels') as f:
    labels = [line.rstrip() for line in f.readlines()]

magpie = Magpie(
    keras_model='current_model/model.h5',
    word2vec_model='current_model/embedding.pkl',
    scaler='current_model/scaler.pkl',
    labels=labels
)

predicted = magpie.predict_from_text('“Ich denke, Du wirst die Scheibe irgendwo innerhalb dieses Kreises treffen”.')
print(predicted[:5])
