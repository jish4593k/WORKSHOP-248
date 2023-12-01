from bs4 import BeautifulSoup
import re
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def get_students_information(html_file):
    names = ['Alunos']
    registration_numbers = ['Matrícula']
    classes = ['Turma']
    combination = []

    # Get HTML
    with open(html_file, 'r', encoding='latin-1') as file:
        soup = BeautifulSoup(file, 'html.parser')

    students_table = soup.findAll("table", {"class": "participantes"})[1]

    # Get students information
    for td in students_table.findAll("td", {"valign": "top"}):
        names.append(td.find('strong').text.strip().title())
        registration_numbers.append(''.join(re.findall(r'Matrícula: <em>(\d+)', str(td))))
        classes.append(''.join(re.findall(r'Turma: <em>(\d*\w*)', str(td))))

    # Combine data
    for i in range(0, len(names)):
        combination.append([registration_numbers[i], names[i], classes[i]])

    return combination

def write_to_csv(data, output_file='ListaAlunos.csv'):
    df = pd.DataFrame(data, columns=['Matrícula', 'Alunos', 'Turma'])
    df.to_csv(output_file, encoding='utf-16', sep='\t', index=False, mode='a', header=False)

def train_neural_network(data):
    # Additional feature: Train a simple neural network using Tensorflow and Keras
    # Assume a simple dataset for demonstration
    # You can replace this with your own dataset and model
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    print("Neural network training completed.")

def prepare_data(data):
    # Additional feature: Prepare dummy data for the neural network
    # Replace this function with your own data preparation logic
    # For demonstration purposes, we assume X and y are features and labels
    X = tf.random.normal((len(data), 10))  # Assume 10 features for each student
    y = tf.random.uniform((len(data), 1), 0, 2, dtype=tf.int32)  # Binary labels (0 or 1)
    return X, y

def build_model():
    # Additional feature: Build a simple neural network model using Keras
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(html_file):
    students_data = get_students_information(html_file)
    write_to_csv(students_data)

    print("Arquivo Criado com Sucesso !")
    print(f"{len(students_data)} alunos matriculados.")

    train_neural_network(students_data)

if __name__ == "__main__":
    html_file_path = sys.argv[1]
    main(html_file_path)
