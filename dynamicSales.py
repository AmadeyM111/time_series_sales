import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit
import os
from typing import Dict, List, Optional, Tuple, Union, Any

# Константы
START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
VISUALIZATION_ENABLED = True  # Флаг для включения/отключения визуализации
PLOT_DIR = '/Users/antonamadeus/Documents/my_project/results/plots/'
RESULTS_DIRECTORY = '/Users/antonamadeus/Documents/my_project/results/'

# Настройка воспроизводимости результатов
np.random.seed(RANDOM_SEED)


def save_plot(fig, filename, directory = PLOT_DIR):
    try:
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        fig.savefig(full_path)
        print(f"График сохранен в {full_path}")
    except Exception as e:
        print(f"Ошибка при сохранении графика: {e}")

# Генерация синтетических данных о продажах с трендом, сезонностью и шумом.
def generate_synthetic_data(start_date=START_DATE, end_date=END_DATE, freq='D') -> pd.DataFrame:
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Компоненты временного ряда
    trend = 0.1 * np.arange(len(dates))  # Линейный тренд
    seasonality = 50 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Годовая сезонность
    noise = np.random.normal(0, 10, len(dates))  # Случайный шум
    
    # Базовое значение + тренд + сезонность + шум
    sales = 1000 + trend + seasonality + noise
    
    return pd.DataFrame({'date': dates, 'sales': sales}).set_index('date')

# Определение оптимального периода сезонности с помощью автокорреляции.
def find_optimal_seasonal_period(data:pd.DataFrame, column: str = 'sales', max_lag: int = 365) -> int:
    
    # Вычисление автокорреляционной функции
    acf_values = acf(data[column].dropna(), nlags=max_lag)
    
    if VISUALIZATION_ENABLED:
        plt.figure(figsize=(12, 6))
        plt.plot(acf_values)
        plt.title('Автокорреляционная функция')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='red')
        plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='red')
        plt.xlabel('Лаг')
        plt.ylabel('Автокорреляция')
        plt.grid(True)

        fig = plt.gcf() # Получить текущую фигуру
        save_plot(fig, 'acf_plot.png')
        if VISUALIZATION_ENABLED:
            plt.show()
    
    # Поиск первого значимого пика после лага 1
    # (исключаем лаг 0, который всегда равен 1)
    significance_level = 1.96 / np.sqrt(len(data))
    for i in range(7, max_lag):  # Начинаем с недельного лага
        if acf_values[i] > significance_level and acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            return i
    
    # Если явных пиков не найдено, возвращаем стандартные периоды
    if data.index.freq == 'D':
        return 7  # Недельная сезонность для дневных данных
    elif data.index.freq == 'M':
        return 12  # Годовая сезонность для месячных данных
    else:
        return 30  # Месячная сезонность по умолчанию
    


# Декомпозиция временного ряда на тренд, сезонность и остаток.
def decompose_time_series(data: pd.DataFrame, column: str = 'sales', period: Optional[int] = None) -> Tuple[Optional[pd.Series],Optional[pd.Series],Optional[pd.Series]]:
    try:
        if period is None:
            period = find_optimal_seasonal_period(data, column)
        
        print(f"Используемый период сезонности: {period}")
    
        # Выполнение декомпозиции
        decomposition = seasonal_decompose(data[column], model='additive', period=period)
        
        if VISUALIZATION_ENABLED:
            # Визуализация результатов декомпозиции
            fig, ( ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10))
            
            ax1.plot(data[column])
            ax1.set_title('Исходный временной ряд')
            ax1.grid(True)
            
            ax2.plot(decomposition.trend)
            ax2.set_title('Тренд')
            ax2.grid(True)
            
            ax3.plot(decomposition.seasonal)
            ax3.set_title('Сезонность')
            ax3.grid(True)
            
            ax4.plot(decomposition.resid)
            ax4.set_title('Остаток')
            ax4.grid(True)
            
            plt.tight_layout()

            # сохранение результата в виде изображения  
            os.makedirs(PLOT_DIR, exist_ok=True)    
            
            plt.savefig(os.path.join(PLOT_DIR, 'decomposition_plot.png'))
            plt.show()
        
        return decomposition.trend, decomposition.seasonal, decomposition.resid
    except Exception as e:
        print(f"Ошибка при декомпозиции временного ряда {e}")
        # Возвращаем None или используем альтернативный подход
        return None, None, None

# Подготовка признаков для моделей машинного обучения.
def prepare_features(data: pd.DataFrame, target_column: str = 'sales', lags: int = 7, add_date_features: bool = True) -> pd.DataFrame:
    features_df = data.copy()

    # Добавление лаговых признаков
    for lag in range(1, lags + 1):
        features_df[f'{target_column}_lag_{lag}'] = features_df[target_column].shift(lag)

    # Добавление признаков даты
    if add_date_features:
        features_df['dayofweek'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['year'] = features_df.index.year
        features_df['dayofyear'] = features_df.index.dayofyear
        
        # Создание циклических признаков для периодических величин
        features_df['dayofweek_sin'] = np.sin(2 * np.pi * features_df.index.dayofweek / 7)
        features_df['dayofweek_cos'] = np.cos(2 * np.pi * features_df.index.dayofweek / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df.index.month / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df.index.month / 12)
    
    # Удаление строк с пропущенными значениями
    features_df = features_df.dropna()
    
    return features_df

# Разделение данных на обучающую и тестовую выборки с учетом временной структуры.
def split_train_test(data: pd.DataFrame, train_ratio: float = TRAIN_TEST_SPLIT_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_size = int(len(data) * train_ratio)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    
    print(f"Размер обучающей выборки: {len(train)}, тестовой выборки: {len(test)}")
    
    return train, test

# Подготовка данных для обучения и тестирования моделей.
def prepare_model_data(train: pd.DataFrame, test: pd.DataFrame, target_column: str = 'sales', feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    if feature_columns is None:
        # Исключаем целевую переменную из признаков
        feature_columns = [col for col in train.columns if col != target_column]
    
    X_train = train[feature_columns]
    y_train = train[target_column]
    
    X_test = test[feature_columns]
    y_test = test[target_column]
    
    return X_train, y_train, X_test, y_test

# Обучение модели XGBoost с оптимизированными параметрами.
def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None) -> XGBRegressor:
   
    if params is None:
        params = {
            'n_estimators': 100,     # Количество деревьев
            'learning_rate': 0.1,    # Скорость обучения
            'max_depth': 5,          # Максимальная глубина деревьев
            'min_child_weight': 1,   # Минимальный вес дочернего узла
            'subsample': 0.8,        # Доля выборки для каждого дерева
            'colsample_bytree': 0.8, # Доля признаков для каждого дерева
            'objective': 'reg:squarederror',  # Функция потерь
            'random_state': RANDOM_SEED
        }
    
    # Создание и обучение модели
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Вывод важности признаков
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Важность признаков XGBoost:")
    print(feature_importance.head(10))
    
    if VISUALIZATION_ENABLED:
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.title('Важность признаков XGBoost')
        plt.xlabel('Важность')
        plt.gca().invert_yaxis()  # Инвертируем ось Y для отображения самых важных признаков сверху
        plt.tight_layout()
        save_plot(plt.gcf(), 'feature_importance_plot.png')

        if VISUALIZATION_ENABLED:
            plt.show()
    
    return model

# Автоматический подбор оптимальных параметров для модели ARIMA.
def find_optimal_arima_params(train_data: pd.Series, seasonal: bool = True, seasonal_period: Optional[int] = None) -> Tuple[Tuple[int, int, int], Optional[Tuple [int, int, int, int]]]:
    
    print("Поиск оптимальных параметров ARIMA...")
    
    # Если сезонный период не указан, используем стандартные значения
    if seasonal and seasonal_period is None:
        if train_data.index.freq == 'D':
            seasonal_period = 7  # Недельная сезонность для дневных данных
        elif train_data.index.freq == 'M':
            seasonal_period = 12  # Годовая сезонность для месячных данных
        else:
            seasonal_period = 30  # Месячная сезонность по умолчанию
    
    # Автоматический подбор параметров
    auto_arima = pm.auto_arima(
        train_data,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,           # Автоматическое определение d
        seasonal=seasonal,
        m=seasonal_period if seasonal else 1,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None if seasonal else 0,  # Автоматическое определение D для сезонной модели
        trace=True,        # Вывод процесса подбора
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,     # Пошаговый поиск для ускорения
        random_state=RANDOM_SEED
    )
    
    # Вывод оптимальных параметров
    if seasonal:
        print(f"Оптимальные параметры SARIMA: {auto_arima.order} x {auto_arima.seasonal_order}")
        return auto_arima.order, auto_arima.seasonal_order
    else:
        print(f"Оптимальные параметры ARIMA: {auto_arima.order}")
        return auto_arima.order, None

#     Обучение модели ARIMA/SARIMA с оптимальными параметрами.
def train_arima_model(train_data: pd.Series, order: Optional[int] = None, seasonal_order: Optional[int] = None):
    try:
        # Если параметры не указаны, используем стандартные
        if order is None:
            order = (1, 1, 1)
        
        # Создание и обучение модели
        if seasonal_order is not None:
            model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
            model_name = "SARIMA"
        else:
            model = ARIMA(train_data, order=order)
            model_name = "ARIMA"
        
        print(f"Обучение модели {model_name}...")
        result = model.fit()
        
        # Вывод сводки модели
        print(result.summary())
        
        return result
    except Exception as e:
        print(f"Ошибка при обучении модели ARIMA: {e}")
        # Возвращаем None или используем запасной вариант модели
        return None

#     Оценка качества моделей по нескольким метрикам.
def evaluate_models(y_true: pd.Series, predictions_dict: Dict[str, np.ndarray], prefix: str = "") -> pd.DataFrame:
   
    metrics = {}
    
    for model_name, y_pred in predictions_dict.items():
        # Расчет метрик
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE может давать ошибку при делении на 0, обрабатываем этот случай
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            mape = np.nan
        
        metrics[model_name] = {
            f'{prefix}RMSE': rmse,
            f'{prefix}MAE': mae,
            f'{prefix}MAPE (%)': mape
        }
    
    # Создание DataFrame с метриками
    metrics_df = pd.DataFrame(metrics).T
    
    # Вывод метрик
    print(f"\n{prefix} Метрики оценки моделей:")
    print(metrics_df)
    
    return metrics_df

# Визуализация фактических значений и прогнозов моделей.
def visualize_predictions(y_true: pd.Series, predictions_dict: Dict[str, np.ndarray], title: str ="Сравнение прогнозов моделей") -> None:
   
    if not VISUALIZATION_ENABLED:
        return
    
    plt.figure(figsize=(15, 7))
    
    # Построение фактических значений
    plt.plot(y_true.index, y_true.values, 'k-', linewidth=2, label='Фактические данные')
    
    # Построение прогнозов для каждой модели
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        plt.plot(y_true.index, y_pred, f'{colors[i % len(colors)]}-', linewidth=1.5, label=f'Прогноз {model_name}')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt.gcf(), 'prediction_plot.png')

    if VISUALIZATION_ENABLED:
        plt.show()

#     Временная кросс-валидация моделей.
def cross_validate_models(data: pd.DataFrame, target_column: str = 'sales', n_splits: int =5, feature_columns: Optional[List[str]] = None) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
   
    if feature_columns is None:
        # Исключаем целевую переменную из признаков
        feature_columns = [col for col in data.columns if col != target_column]
    
    # Создание объекта временной кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Словари для хранения результатов
    cv_results = {
        'XGBoost': {'RMSE': [], 'MAE': [], 'MAPE': []},
        'ARIMA': {'RMSE': [], 'MAE': [], 'MAPE': []}
    }
    
    # Выполнение кросс-валидации
    for i, (train_idx, test_idx) in enumerate(tscv.split(data)):
        print(f"\nФолд {i+1}/{n_splits}")
        
        # Разделение данных
        cv_train = data.iloc[train_idx]
        cv_test = data.iloc[test_idx]
        
        # Подготовка данных для XGBoost
        X_train = cv_train[feature_columns]
        y_train = cv_train[target_column]
        X_test = cv_test[feature_columns]
        y_test = cv_test[target_column]
        
        # Обучение и прогноз XGBoost
        xgb_model = train_xgboost_model(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Обучение и прогноз ARIMA
        # Для простоты используем фиксированные параметры в кросс-валидации
        arima_model = ARIMA(cv_train[target_column], order=(1, 1, 1))
        arima_result = arima_model.fit()
        arima_pred = arima_result.forecast(steps=len(cv_test))
        
        # Оценка моделей
        predictions = {
            'XGBoost': xgb_pred,
            'ARIMA': arima_pred
        }
        
        # Расчет метрик
        for model_name, y_pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            try:
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            except:
                mape = np.nan
            
            cv_results[model_name]['RMSE'].append(rmse)
            cv_results[model_name]['MAE'].append(mae)
            cv_results[model_name]['MAPE'].append(mape)
    
    # Вычисление средних значений метрик
    cv_summary = {}
    for model_name, metrics in cv_results.items():
        cv_summary[model_name] = {
            'Avg RMSE': np.mean(metrics['RMSE']),
            'Avg MAE': np.mean(metrics['MAE']),
            'Avg MAPE (%)': np.mean([m for m in metrics['MAPE'] if not np.isnan(m)])
        }
    
    # Вывод результатов кросс-валидации
    cv_summary_df = pd.DataFrame(cv_summary).T
    print("\nРезультаты кросс-валидации:")
    print(cv_summary_df)
    
    return cv_results, cv_summary_df

# Сохранение графика в указанную директорию с обработкой ошибок.

def save_results(metrics_df: pd.DataFrame, 
                cv_summary: pd.DataFrame, 
                directory: str = RESULTS_DIRECTORY) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        metrics_df.to_csv(os.path.join(directory, 'model_metrics.csv'))
        cv_summary.to_csv(os.path.join(directory, 'cv_summary.csv'))
        print(f"Результаты сохранены в {directory}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")


# Основная функция для выполнения всего процесса прогнозирования.
def main():
        print("Начало процесса прогнозирования временных рядов")
        
        # 1. Генерация синтетических данных
        print("\n1. Генерация синтетических данных")
        data = generate_synthetic_data()
        print(f"Сгенерированы данные с {len(data)} наблюдениями")
        print(data.head())
        
        # 2. Декомпозиция временного ряда
        print("\n2. Декомпозиция временного ряда")
        trend, seasonal, residual = decompose_time_series(data)
        
        # Добавление компонентов в DataFrame
        data['trend'] = trend
        data['seasonal'] = seasonal
        data['residual'] = residual
        data = data.dropna()  # Удаление строк с NaN после декомпозиции
        
        # 3. Подготовка признаков
        print("\n3. Подготовка признаков")
        features_data = prepare_features(data)
        print(f"Подготовлены данные с {len(features_data)} наблюдениями и {features_data.shape[1]} признаками")
        print(features_data.head())
        
        # 4. Разделение на обучающую и тестовую выборки
        print("\n4. Разделение на обучающую и тестовую выборки")
        train, test = split_train_test(features_data)
        
        # 5. Подготовка данных для моделей
        print("\n5. Подготовка данных для моделей")
        # Исключаем целевую переменную и некоторые компоненты из признаков
        feature_columns = [col for col in features_data.columns 
                        if col not in ['sales', 'residual']]
        
        X_train, y_train, X_test, y_test = prepare_model_data(
            train, test, target_column='sales', feature_columns=feature_columns
        )
        
        # 6. Обучение и оценка модели XGBoost
        print("\n6. Обучение и оценка модели XGBoost")
        xgb_model = train_xgboost_model(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # 7. Поиск оптимальных параметров ARIMA
        print("\n7. Поиск оптимальных параметров ARIMA")
        arima_order, seasonal_order = find_optimal_arima_params(
            train['sales'], seasonal=True
        )
        
        # 8. Обучение и оценка модели ARIMA
        print("\n8. Обучение и оценка модели ARIMA")
        arima_result = train_arima_model(
            train['sales'], order=arima_order, seasonal_order=seasonal_order
        )
        arima_pred = arima_result.forecast(steps=len(test))
        
        # 9. Оценка моделей
        print("\n9. Оценка моделей")
        predictions = {
            'XGBoost': xgb_pred,
            'ARIMA': arima_pred
        }
        
        metrics_df = evaluate_models(y_test, predictions, prefix="Test ")
        
        # 10. Визуализация результатов
        print("\n10. Визуализация результатов")
        visualize_predictions(y_test, predictions)
        
        # 11. Кросс-валидация моделей
        print("\n11. Кросс-валидация моделей")
        cv_results, cv_summary = cross_validate_models(
            features_data, target_column='sales', n_splits=3, feature_columns=feature_columns
        )

        # Сохранение результатов
        save_results(metrics_df, cv_summary)
    
        # Возврат результатов
        return {
            'data': data,
            'train': train,
            'test': test,
            'xgb_model': xgb_model,
            'arima_result': arima_result,
            'predictions': predictions,
            'metrics': metrics_df,
            'cv_summary': cv_summary
        }
        
if __name__ == "__main__":
    results = main()


