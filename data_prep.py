import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

def load_data(filepath):
    """Carrega os dados do arquivo CSV."""
    print(f"Carregando dados de {filepath}...")
    return pd.read_csv(filepath)

def preprocess_features(df):
    print("Pré-processando features...")
    
    # Extrair informações da data de venda
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.drop('date', axis=1)
    df = df.drop('id', axis=1)
    df['house_age'] = df['year'] - df['yr_built']
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['basement_ratio'] = df['sqft_basement'] / df['sqft_living']
    df['basement_ratio'] = df['basement_ratio'].fillna(0)
    df['price_log'] = np.log(df['price'])
    return df

def split_data(df, test_size=0.2, random_state=42):
    print(f"Dividindo dados em treino ({1-test_size:.0%}) e teste ({test_size:.0%})...")
    X = df.drop(['price', 'price_log'], axis=1)
    y = df['price_log'] 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                        'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year', 'month',
                        'day_of_week', 'house_age', 'basement_ratio']
    
    categorical_features = ['waterfront', 'view', 'condition', 'grade', 'zipcode', 'renovated']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def main():
    df = load_data('dataset/kc_house_data.csv')
    df = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = create_preprocessing_pipeline()
    print("Ajustando o pipeline de pré-processamento nos dados de treino...")
    preprocessor.fit(X_train)
    
    print("Transformando os dados de treino e teste...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Salvando dados processados e preprocessador...")
    joblib.dump(preprocessor, 'processed_data/preprocessor.pkl')
    
    X_train.to_csv('processed_data/X_train_original.csv', index=False)
    X_test.to_csv('processed_data/X_test_original.csv', index=False)
    
    y_train.to_csv('processed_data/y_train.csv', index=False)
    y_test.to_csv('processed_data/y_test.csv', index=False)
    
    np.save('processed_data/X_train_processed.npy', X_train_processed)
    np.save('processed_data/X_test_processed.npy', X_test_processed)
    np.save('processed_data/y_train.npy', y_train.values)
    np.save('processed_data/y_test.npy', y_test.values)
    
    with open('processed_data/feature_names.txt', 'w') as f:
        f.write(str(X_train.columns.tolist()))
    
    print(f"Dados processados salvos em: 'processed_data/'")
    print(f"Shape dos dados de treino: {X_train_processed.shape}")
    print(f"Shape dos dados de teste: {X_test_processed.shape}")

if __name__ == "__main__":
    main() 