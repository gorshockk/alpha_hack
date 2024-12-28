import pandas as pd
from sklearn.decomposition import IncrementalPCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import f1_score
import multiprocessing

# Параметры для уменьшения размерности
n_components = 75
batch_size = 10000
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# Путь к файлу
data_path = '/content/drive/MyDrive/train_10.csv'

# Подгонка PCA по частям
for chunk in pd.read_csv(data_path, chunksize=batch_size):
    ipca.partial_fit(chunk.drop(['target', 'smpl', 'id'], axis=1))

# Список новых названий признаков
new_columns = [f'feature_{i}' for i in range(1, n_components + 1)]

# Функция для сжатия данных с помощью IncrementalPCA
def compression(filename, features=['target', 'smpl', 'id']):
    results = []
    for chunk in pd.read_csv(filename, chunksize=batch_size):
        # Сохраняем базовые данные
        base_info = chunk[features].reset_index(drop=True)
        # Применяем IncrementalPCA к признакам
        transformed_data = pd.DataFrame(ipca.transform(chunk.drop(features, axis=1)))
        # Объединяем базовую информацию и новые признаки
        result = pd.concat([base_info, transformed_data], axis=1)
        result.columns = [*features, *new_columns]
        # Добавляем обработанный кусок в список
        results.append(result)
    # Объединяем все обработанные части в один DataFrame
    return pd.concat(results, ignore_index=True)

# Применяем функцию к файлу
compressed_data = compression(data_path)
compressed_data.to_csv('train_compressed.csv', index=False)

# Чтение train
data_train = pd.read_csv(data_path)
data_train = data_train[[col for col in data_train.columns if not col.startswith('Unnamed:')]]
data_train = data_train.copy()

# Разделение на признаки и целевую переменную
target = data_train['target']  # Убедитесь, что 'target' есть в данных
data_train = data_train.drop(['target', 'smpl', 'id'], axis=1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data_train, target, test_size=0.7, stratify=target, random_state=42)

# Гиперпараметры с максимальными значениями
params = {
    'n_estimators': 3500,
    'learning_rate': 0.01,
    'max_depth': 11,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'max_leaves': 10,
    'reg_alpha': 0.8,
    'scale_pos_weight': sum(target == 0) / sum(target == 1),
    'random_state': 42,
    'n_jobs': multiprocessing.cpu_count(),
    'eval_metric': 'logloss',
    'tree_method': 'hist'  # Используем метод гистограммы
    # 'device': 'cuda'       # Указываем использование GPU через CUDA (если доступно)
}

# Создаем модель с максимальными параметрами
model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Предсказания и оценка модели
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
