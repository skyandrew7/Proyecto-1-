# Proyecto-1-
##Primer proyecto integrador henry MLOPS

## DATOS 
Se toman los datos de https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj.

##OBJETIVOS DEL PROYECTO
1.- Desarrollar 5 funciones de busquedad enlasadas a los tres datasets, es decir identificacion de sus uniones

2.- Preparar un modelo de sgerencia en base a la data proporcionada.
## Inicio del proyecto-1
Al ser 3 datasets se hace un analisis individual de cada dataset por separado.

##Primer dataset outpusteamgames:

##ETL

![image](https://github.com/user-attachments/assets/b7750e98-a460-4386-a91d-365f42243df6)

+ Se observan varios errores de tipo de datos ademas de un dataset con muchas dimensiones.

+ Se realiza la eliminacion de las columnas irrevalentes: url, early access, reviews url, y specs.

+ Se eliminan valores duplicados y valores nulos correspondietes al dataset.

##EDA

![image](https://github.com/user-attachments/assets/6e475b1e-f6fe-4f7c-bf9c-11ae1809d247)


Podemos observar la presencia datos duplicados, pero con cambios tipograficos.
Un dato muy interesante es apreciar la cantidad de publicadores que solo poseen un juego. 

##segudo dataset user australian review

El segundo dataset propone mas un reto logico que un reto analitico a primera vista, porque solo cuenta con tres columnas principales, sin embargo una de esas
columnas es una columna anidada con mucha mas informacion. 

##Tercer dataset user australian items

Al igual que el segundo dataset, el primer reto al tratar este dataset es sus columnas anidadas en formato json. 
##ETL 

+ Eliminamos columnas irrelevantes y hacemos una primera limpieza de valores nulos. 
+ Se realiza la conversion necesaria de las columnas. 


## Analisis de sentimientos 

Como requisito para cumplir uno de los objetivos se nos pide realizar un analisis de sentimientos 
en funcion de las reviews que han hecho los usuarios. 

DEsde mi enfoque, decidir hacer dos pruebas del modelo que cree, una sin usar herramientas de nlp 
y otra usando herramientas de nlp. El resultado es muy llamativo. 

Dejo el codigo para permitir su replica y comparacion. Me parecio muy interesante. 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


## Función para convertir la recomendación (True/False) en valores de 0 (negativo) o 2 (positivo)
def map_recommend_to_sentiment(recommend):
    return 2 if recommend else 0

## Crear una nueva columna con las recomendaciones convertidas a sentimientos (0 = no recomienda, 2 = recomienda)
df_reviews['true_sentiment_from_recommend'] = df_reviews['recommend'].apply(map_recommend_to_sentiment)

## Filtrar solo las reseñas donde el sentimiento es negativo o positivo (ignorar los neutrales si así lo decides)
df_filtered = df_reviews[df_reviews['sentiment_analysis2'] != 1]

## Variables para los valores reales y predichos (sin neutrales)
y_true = df_filtered['true_sentiment_from_recommend']
y_pred = df_filtered['sentiment_analysis2']

## Evaluar las métricas de rendimiento
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

## Matriz de Confusión
conf_matrix = confusion_matrix(y_true, y_pred)

## Visualización de la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 2], yticklabels=[0, 2])
plt.xlabel('Predicted Sentiment')
plt.ylabel('True Sentiment from Recommend')
plt.title('Confusion Matrix')
plt.show()

##Modelo de recomendacion
Este algoritmo de recomendacion se lo realiza en base a 'cosine similaryty'. Para realizar este algoritmo se 
realizo un enfoque item-item es decir el usuario id del juego y el sistema le recomienda juegos similares.

Nuevamente se deja el codigo para comparacion y experimentacion. 

# Convertir los géneros a un formato numérico binario usando one-hot encoding
generos_matrix = df_juegos_sample['generos del juego'].str.get_dummies(sep=',')
juegos_matrix = generos_matrix.values

# Calcular la similitud de coseno entre los juegos
cosine_sim = cosine_similarity(juegos_matrix)

# Función para obtener recomendaciones solo con item_id
def recomendar_juegos(item_id, top_n=5):
    # Encontrar el índice del juego con el item_id dado
    if item_id not in df_juegos_sample['item_id'].values:
        return f"Item ID {item_id} no encontrado en los datos."
    
    idx = df_juegos_sample[df_juegos_sample['item_id'] == item_id].index[0]
    
    # Calcular la similitud con todos los demás juegos
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar juegos basados en la similitud del coseno
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener los índices de los juegos más similares (sin contar el mismo juego)
    sim_scores = sim_scores[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    
    # Retornar los juegos recomendados
    return df_juegos_sample.iloc[recommended_indices][['nombre_del_juego', 'item_id']]

# Ejemplo de recomendación (entrada solo item_id)
item_id_example = 12345  # Reemplaza con un item_id válido de tu dataset
print(recomendar_juegos(item_id_example))

# Validación cruzada con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = juegos_matrix
y = df_reviews_sample['recommend']
