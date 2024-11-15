
-- создаем схему данных для ML
CREATE SCHEMA IF NOT EXISTS ml;

-- создаем таблицу сырых данных для обучения
CREATE TABLE IF NOT EXISTS ml.raw_train_data (
	customer_id integer NOT NULL,
	surname varchar(30) NOT NULL,
	credit_score smallint NOT NULL,
	geography varchar(20) NOT NULL,
	gender varchar(10) NOT NULL,
	age smallint NOT NULL,
	tenure smallint NOT NULL,
	balance numeric(12,2) NOT NULL,
	num_of_products smallint NOT NULL,
	has_cr_card smallint NOT NULL,
	is_active_member smallint NOT NULL,
	estimated_salary numeric(12,2) NOT NULL,
	exited smallint NOT NULL
);

-- создаем таблицу сырых данных для прогнозирования
CREATE TABLE IF NOT EXISTS ml.raw_test_data (
	customer_id integer NOT NULL,
	surname varchar(30) NOT NULL,
	credit_score smallint NOT NULL,
	geography varchar(20) NOT NULL,
	gender varchar(10) NOT NULL,
	age smallint NOT NULL,
	tenure smallint NOT NULL,
	balance numeric(12,2) NOT NULL,
	num_of_products smallint NOT NULL,
	has_cr_card smallint NOT NULL,
	is_active_member smallint NOT NULL,
	estimated_salary numeric(12,2) NOT NULL
);

-- создаем таблицу подготовленных данных для обучения
-- здесь столбцы customer_id и surname не важны - их можно отбросить на этапе подготовки
CREATE TABLE IF NOT EXISTS ml.x_train (
	credit_score smallint NOT NULL,
	geography smallint NOT NULL,
	gender smallint NOT NULL,
	age smallint NOT NULL,
	tenure smallint NOT NULL,
	balance numeric(12,2) NOT NULL,
	num_of_products smallint NOT NULL,
	has_cr_card smallint NOT NULL,
	is_active_member smallint NOT NULL,
	estimated_salary numeric(12,2) NOT NULL,
	exited smallint NOT NULL
);

-- создаем таблицу подготовленных данных прогнозирования
-- здесь важно не потерять идентификатор клиента, чтобы прогнозы можно было соотнести
CREATE TABLE IF NOT EXISTS ml.x_test (
	customer_id integer NOT NULL,
	credit_score smallint NOT NULL,
	geography smallint NOT NULL,
	gender smallint NOT NULL,
	age smallint NOT NULL,
	tenure smallint NOT NULL,
	balance numeric(12,2) NOT NULL,
	num_of_products smallint NOT NULL,
	has_cr_card smallint NOT NULL,
	is_active_member smallint NOT NULL,
	estimated_salary numeric(12,2) NOT NULL
);

-- создаем таблицу результатов прогнозирования
CREATE TABLE IF NOT EXISTS ml.predictions (
	customer_id integer NOT NULL,
	credit_score smallint NOT NULL,
	geography smallint NOT NULL,
	gender smallint NOT NULL,
	age smallint NOT NULL,
	tenure smallint NOT NULL,
	balance numeric(12,2) NOT NULL,
	num_of_products smallint NOT NULL,
	has_cr_card smallint NOT NULL,
	is_active_member smallint NOT NULL,
	estimated_salary numeric(12,2) NOT NULL,
	proba_0 real NOT NULL,
	proba_1 real NOT NULL
);

-- создаем таблицу для скоринга обученных моделей
CREATE TABLE IF NOT EXISTS ml.model_score (
	train_id serial PRIMARY KEY,
	model_params text,
	score_name varchar(30),
	score_value real,
	trained_at timestamp DEFAULT now()
);

-- подключаем расширение pl/python для использования python в процедурах postgres
CREATE EXTENSION IF NOT EXISTS plpython3u;

-- загружаем сырые данные обучающей и проверочной выборок в соответствующие таблицы
COPY ml.raw_train_data FROM '/home/ml/data/raw_train_data.csv' WITH (FORMAT csv, HEADER true);
COPY ml.raw_test_data FROM '/home/ml/data/raw_test_data.csv' WITH (FORMAT csv, HEADER true);

-- создаем процедуру для подготовки обучающей выборки
CREATE OR REPLACE FUNCTION ml.make_x_train()
RETURNS VOID
AS $$

	import os
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder
	from imblearn.over_sampling import ADASYN
	from sqlalchemy import create_engine

	# получаем сырые данные обучающей выборки
	result = plpy.execute('SELECT * FROM ml.raw_train_data')
	data = [dict(record) for record in result]
	df = pd.DataFrame(data)

	# удаляем неинформативные столбцы
	df = df.drop(columns=['customer_id', 'surname'])
	
	# кодируем столбцы строковых переменных с помощью LabelEncoder
	le = LabelEncoder()
	df['gender'] = le.fit_transform(df['gender'])
	df['geography'] = le.fit_transform(df['geography'])
	
	# выделяем пространство признаков и целевую переменную
	X = df.drop(columns=['exited'])
	y = df['exited']
	
	# устраняем дисбаланс классов
	adasyn = ADASYN(random_state=42)
	X_resampled, y_resampled = adasyn.fit_resample(X, y)
	X_resampled['exited'] = y_resampled
	
	# сохраняем результат в таблицу базы данных
	engine = create_engine(f'postgresql+psycopg2://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@127.0.0.1:5432/{os.environ["POSTGRES_DB"]}')
	X_resampled.to_sql('x_train', engine, schema='ml', if_exists='replace', index=False)
	
$$ LANGUAGE plpython3u;

-- создаем процедуру для подготовки проверочной выборки
CREATE OR REPLACE FUNCTION ml.make_x_test()
RETURNS VOID
AS $$

	import os
	import pandas as pd
	from sklearn.preprocessing import LabelEncoder
	from sqlalchemy import create_engine

	# получаем сырые данные проверочной выборки
	result = plpy.execute('SELECT * FROM ml.raw_test_data')
	data = [dict(record) for record in result]
	df = pd.DataFrame(data)

	# удаляем неинформативные столбцы, но оставляем идентификатор клиента
	df = df.drop(columns=['surname'])

	# кодируем столбцы строковых переменных с помощью LabelEncoder
	le = LabelEncoder()
	df['gender'] = le.fit_transform(df['gender'])
	df['geography'] = le.fit_transform(df['geography'])

	# сохраняем результат в таблицу базы данных
	engine = create_engine(f'postgresql+psycopg2://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@127.0.0.1:5432/{os.environ["POSTGRES_DB"]}')
	df.to_sql('x_test', engine, schema='ml', if_exists='replace', index=False)

$$ LANGUAGE plpython3u;

-- создаем процедуру для обучения и оптимизации модели 
CREATE OR REPLACE FUNCTION ml.make_model()
RETURNS VOID
AS $$

	import os
	import pickle
	import optuna
	import pandas as pd
	
	from io import BytesIO
	from datetime import datetime
	from sklearn.model_selection import cross_val_score
	from catboost import CatBoostClassifier
	from optuna.samplers import TPESampler
	from functools import partial

	# получаем подготовленные данные обучающей выборки
	result = plpy.execute('SELECT * FROM ml.x_train')
	data = [dict(record) for record in result]
	df = pd.DataFrame(data)

	# формируем набор признаков и целевой переменной
	X = df.drop(columns=['exited'])
	y = df['exited']

	# пишем функцию оптимизации optuna для модели catboost
	def objective(trial: optuna.Trial, X_train, y_train):
	    params = {
	        'iterations': trial.suggest_int('iterations', 100, 1000),
	        'depth': trial.suggest_int('depth', 4, 10),
	        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
	        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
	        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
	        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
	        'border_count': trial.suggest_int('border_count', 32, 255),
	        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
	    }
	    # инициализируем классификатор и считаем оценку на перекрестной проверке
	    catboost_model = CatBoostClassifier(**params, silent=True)
	    score = cross_val_score(catboost_model, X_train, y_train, cv=3, scoring='accuracy').mean()
	
	    return score

	# создаем сессию оптимизации optuna
	sampler = TPESampler(seed=42)
	study = optuna.create_study(sampler=sampler, direction='maximize')
	study.optimize(partial(objective, X_train=X, y_train=y), n_trials=5)

	# сохраняем скоринг и параметры лучшей модели в таблицу
	stmt = """
		INSERT INTO ml.model_score (model_params, score_name, score_value)
		VALUES ($1, 'accuracy', $2)
	"""
	pr_stmt = plpy.prepare(stmt, ['text', 'real'])
	plpy.execute(pr_stmt, [str(study.best_params), study.best_value])

	# строим модель catboost с лучшими параметрами оптимизации
	params = {
	    'iterations': study.best_params['iterations'],
	    'depth': study.best_params['depth'],
	    'learning_rate': study.best_params['learning_rate'],
	    'l2_leaf_reg': study.best_params['l2_leaf_reg'],
	    'random_strength': study.best_params['random_strength'],
	    'bagging_temperature': study.best_params['bagging_temperature'],
	    'border_count': study.best_params['border_count'],
	    'boosting_type': study.best_params['boosting_type']
	}
	best_catboost_model = CatBoostClassifier(**params, silent=True)
	
	# обучаем модель на данных для обучения
	best_catboost_model.fit(X, y, silent=True)
	
	# сохраняем обученную модель на диск
	model_data = BytesIO()
	pickle.dump(best_catboost_model, model_data)
	model_data.seek(0)
	
	with open(f'{os.environ["PGDATA"]}/catboost_model.pkl','wb') as f:
	    f.write(model_data.read())

$$ LANGUAGE plpython3u;

-- создаем процедуру для получения прогнозов на проверочной выборке
CREATE OR REPLACE FUNCTION ml.make_predictions()
RETURNS VOID
AS $$

	import os
	import pickle
	import pandas as pd	
	
	from sqlalchemy import create_engine
	from catboost import CatBoostClassifier

	# загружаем сохраненную обученную модель
	with open(f'{os.environ["PGDATA"]}/catboost_model.pkl', 'rb') as f:
	    model = pickle.load(f)

	# получаем подготовленные данные проверочной выборки
	result = plpy.execute('SELECT * FROM ml.x_test')
	data = [dict(record) for record in result]
	df = pd.DataFrame(data)

	# устанавливаем идентификатор клиента в качетсве индекса
	df = df.set_index('customer_id')

	# передаем данные модели и получаем прогнозы вероятностей классов
	preds = pd.DataFrame(columns=['proba_0', 'proba_1'], data=model.predict_proba(df))

	# добавляем к датафрейму признаков вероятности классов
	predictions = df.copy()
	predictions = predictions.reset_index()
	predictions['proba_0'] = preds['proba_0']
	predictions['proba_1'] = preds['proba_1']
	
	# сохраняем результат в таблицу базы данных
	engine = create_engine(f'postgresql+psycopg2://{os.environ["POSTGRES_USER"]}:{os.environ["POSTGRES_PASSWORD"]}@127.0.0.1:5432/{os.environ["POSTGRES_DB"]}')
	predictions.to_sql('predictions', engine, schema='ml', if_exists='replace', index=False)

$$ LANGUAGE plpython3u;