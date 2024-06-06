from sklearn.datasets import fetch_20newsgroups
import pickle
import os

def fetch_and_save_data():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = {'data': newsgroups.data, 'target': newsgroups.target, 'target_names': newsgroups.target_names}
    with open('data/newsgroups_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    fetch_and_save_data()

