# Deep-Wittgenstein

In this repository we present a pretrained model for classifiying
Wittgenstein's remarks. The pretrained model can detect and classify 70
different categories for a remark: Jetzt, Regel, Sprache, Gedanke, Behauptung,
Mengenlehre, Gleich, Unendliche, Möglichkeit, Begriff, Idealismus, Gegenstand,
Kardinalzahlen, Phänomenologie, Hypothese, Ursache, Ungefähr, Unendlichkeit,
Entdeckung, Problem, Mathematik, Metamathematik, Schmerzen, Sprache,
Sprachspiel, Satz, Klasse, Erwartung, und, Erfüllung, Gesichtsraum, XXX,
Bedeutung, Grund, Sinn, Philosophie, Versuchen, Suchen, Vorstellung, Abbild,
Fähigkeit, Zeit, Logik, Farben, und, Farbenmischung, Minima, Visibilia, Grund,
des, Denkens, W-F-Notation, Undeutlichkeit, Glaube, Wissen, Logische, Form,
Tabelle, Anwendung, Unmittelbares, Allgemeinheit, Grammatik, Zeichen, Schach,
Folgen, Beweis, Mathematik, Induktion, Induktionsbeweis, Wahrscheinlichkeit,
Gebrauch, Meinen, Physikalischer, Raum, Absicht, Im, selben, Sinn, Zahlen,
Regel, Erfahrungssatz, Nicht, Verifikation, Verstehen, Tonfolge, Physikalische,
Sprache and Denken.

This work was done during summer semester 2017 with support by [Dr. Maximilian
Hadersbeck](http://cis.lmu.de/personen/mitarbeiter/hadersbeck/index.html) ([LMU
Munich](https://www.en.uni-muenchen.de/index.html)). Hand-labeled data is
provided by [Dr. Josef G. F.
Rothhaupt](http://www.philosophie.uni-muenchen.de/lehreinheiten/philosophie_1/personen/josef_rothhaupt/index.html)
([LMU Munich](https://www.en.uni-muenchen.de/index.html)).

This project was funded by
[Lehre@LMU](https://www.uni-muenchen.de/studium/lehre_at_lmu/index.html) with a
NVIDIA Jetson TX-1.

## Example

Input remark:

```text
Der Unterschied der Wortarten ist immer wie der Unterschied der Spielfiguren,
oder, wie der noch größere, einer Spielfigur und des Schachbrettes.
```

Hand-labeled gold label: "Grammatik"

# Requirements

The multi-label classification approach is implemented with *Keras*, *TensorFlow*
and the *magpie* library. The following libraries must be installed:

| Library      | Version (tested)
| ------------ | ----------------
| *magpie*     | 2.0
| *Keras*      | 2.1.3
| *TensorFlow* | 1.5.0
| *h5py*       | 2.7.1

Notice: *magpie* should be installed via:

```bash
pip3 install --user git+https://github.com/inspirehep/magpie.git@v2.0
```

# Dataset

Hand-labeled data is available for the complete Ts-212. Thus, hand-labeled
categories for 7099 remarks are used. Then this corpus is split into training,
development and test set.

| Dataset     | # Remarks
| ----------- | ---------
| Training    | 5620
| Development | 719
| Test        | 760

# Pretrained model

The pretrained model consists of four files:

| Description | Download
| ----------- | --------
| Word Embeddings | [embedding.pkl](https://github.com/stefan-it/deep-wittgenstein/raw/master/current_model/embedding.pkl)
| Model           | [model.h5](https://github.com/stefan-it/deep-wittgenstein/raw/master/current_model/model.h5)
| Scaler          | [scaler.pkl](https://github.com/stefan-it/deep-wittgenstein/raw/master/current_model/scaler.pkl)
| Category labels | [categories.labels](https://github.com/stefan-it/deep-wittgenstein/raw/master/categories.labels)

Word embeddings, model and scaler are located in the `current_model` of this
repository. `categories.labels` is located in the root folder of this repository.

# Classification - Example

To classify new remarks of Ludwig Wittgenstein, the following script can be used:

```python
from magpie import Magpie

with open('categories.labels') as f:
    labels = [line.rstrip() for line in f.readlines()]

magpie = Magpie(
    keras_model='current_model/model.h5',
    word2vec_model='current_model/embedding.pkl',
    scaler='current_model/scaler.pkl',
    labels=labels
)
```

This loaded the pretrained model with all its dependencies like word embeddings
or labels.

Then the following command can be used to classifiy a remark:

```python
predicted = magpie.predict_from_text('“Ich denke, Du wirst die Scheibe irgendwo innerhalb dieses Kreises treffen”.')
print(predicted)
```

This will output of 5 best predicted categories for the input remark:

```json
[('Allgemeinheit', 0.66499853), ('Folgen', 0.53158545),
 ('Regel', 0.004923807), ('Satz', 0.0018804041), ('Meinen', 0.0017680882)]
```

The gold categories are "Allgemeinheit" and "Folgen".

This classification script is located under `classification.py`.

# Acknowledgements

We would like to thank Dr. Maximilian Hadersbeck for his great support during
the development phase. We also want to thank Dr. Josef G. F. Rothhaupt for
providing us high-quality hand-labeled data for over 7000 remarks of Ludwig
Wittgenstein.

We are deeply grateful that Lehre@LMU funded our research project with a
NVIDIA Jetson TX1 developer board and we would like thank LMU Munich for this
awesome program. This really helps students and boosts research.
