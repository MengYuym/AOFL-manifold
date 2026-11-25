import pickle

with open('AudioFeatures.pkl', 'rb') as file:
    data = pickle.load(file)
print(data)