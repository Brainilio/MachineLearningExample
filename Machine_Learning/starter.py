"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""

# READ: UNCOMMENT SECTION AND RUN FOR CODE

# All needed libraries



from machinelearningdata import Machine_Learning_Data 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle


def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


studentnummer = "0910135" # TODO: aanpassen aan je eigen studentnummer

assert studentnummer != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(studentnummer)

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)




# UNSUPERVISED LEARNING

# TODO: print deze punten uit en omcirkel de mogelijke clusters


# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes










# --------------------- K Means Algorithm --------------------- #

# # (UN)COMMENT FROM HERE

# # haal clustering data op
# kmeans_training = data.clustering_training()

# # extract de x waarden
# X = extract_from_json_as_np_array("x", kmeans_training)

# # slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
# x = X[...,0]
# y = X[...,1]

# # ik wil 3 clusters, kmeans++ doet aantal clusters pakken en het op een handige manier, n_init geeft aan hoeveel iteraties je wilt met maximaal 300 iteraties mogelijk
# kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300) 
# kmeans.fit(X)

# # centroids tekenen
# clusters = kmeans.cluster_centers_ 

# # aantal iteraties die zijn gedaan aangeven
# iterations = kmeans.n_iter_ 

# # label voor elk punt 
# labels = kmeans.labels_ 

# # print out clusters / ik wil de punten van de clusters zien en de aantal iteraties die zijn gedaan
# print(clusters, iterations) 



# # dit zijn de kleuren die ik wil geven aan elke cluster
# colors = ["g.", "r.", "c.", "y."] 

# # Loopen door dataset
# for i in range(len(X)): 

#     # ik wil elke cluster een label geven en ik wil alle coordinaten geschreven zien 
#     print("coordinate:", X[i], "label:", labels[i]) 

#     # geef voor elk label een kleur, plot 
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10) 

#     #dit is de centroid aangeven met een X op de map
#     plt.scatter(clusters[:, 0], clusters[:, 1], marker = "x", s=150, linewidths= 5, zorder = 10) 

#     #save de cluster in een bestand (save.pkl)
#     pickle.dump(kmeans, open("save.pkl", "wb")) 
   

# # show en plot alle data
# plt.axis([min(x), max(x), min(y), max(y)]) 
# plt.show()

# # (UN)COMMENT TO HERE









# # SUPERVISED LEARNING

# # TODO: leer de classificaties

# # TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
# #       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
# #       bijvoordeeld Y_predict

# # TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt

# # TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
# #       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
# #       geen echte Y-waarden gekregen hebt.
# #       onderstaande code stuurt je voorspelling naar de server, die je dan
# #       vertelt hoeveel er goed waren.










# # --------------------- Decision Tree --------------------- #

# # (UN)COMMENT FROM HERE

# # Split X's & Y's into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# # Add DecisionTree to Variable
# classifier = DecisionTreeClassifier()

# # Train your X and Ys
# classifier.fit(x_train, y_train)

# # Y predictions to your X
# y_pred = classifier.predict(x_test)

# # Print your accuracy score: Y_test vs y_pred
# print("accuracy:", accuracy_score(y_test, y_pred))

# # Plotting Decision Tree 

# # haal data op om te testen
# classification_test = data.classification_test()
# # testen doen we 'geblinddoekt' je krijgt nu de Y's niet
# X_test = extract_from_json_as_np_array("x", classification_test)

# Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# # stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
# classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
# print("Classificatie accuratie (test): " + str(classification_test))

# # Is het mogelijk om een model op te slaan, een getrained model? hoe weet je hoe goed een model getrained is? hoe frequent moet je een model
# # Blijven trainen tot het een goed model is?
# # Data op een mooiere manier plotten, in een grafiek misschien?

# # (UN)COMMENT TO HERE













# # --------------------- Logistic Regression --------------------- #

# # (UN)COMMENT FROM HERE

# # Add sklearn's regression model to a variable
# regression = LogisticRegression()

# # Split X & Y Data from Classification-training into training and testing variables
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# # Train your model
# regression.fit(x_train, y_train)

# # Predict X-test variable
# prediction = regression.predict(x_test)

# # Print out predictions (Y's to the X's)
# print(prediction)

# # Add accuracy score to a variable 
# score = regression.score(x_test, y_test)

# # Print out accuracy score
# print("accuracy:", score)

# # # Print your accuracy score: Y_test vs y_pred
# # print("accuracy:", accuracy_score(y_test, prediction))

# # Plot outputs
# x_3 = x_test[...,0]
# y_3 = x_test[...,1]

# for i in range(len(x_3)):
#     plt.plot(x_3[i], y_3[i], 'b.') # k = zwart

# plt.axis([min(x_3), max(x_3), min(y_3), max(y_3)])
# plt.show()

# # haal data op om te testen
# classification_test = data.classification_test()
# # testen doen we 'geblinddoekt' je krijgt nu de Y's niet
# X_test = extract_from_json_as_np_array("x", classification_test)

# Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# # stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
# classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
# print("Classificatie accuratie (test): " + str(classification_test))

# # (UN)COMMENT TO HERE