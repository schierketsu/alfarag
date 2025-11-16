## Описание решения

В папке `solution` реализован retrieval-пайплайн (RAG-часть), который:

- индексирует корпус из `techzadanie/websites.csv` (разбиение на чанки, эмбеддинги, FAISS-индекс);
- по вопросам из `techzadanie/questions_clean.csv` находит топ-5 релевантных документов `web_id`;
- формирует файл `submit.csv` в формате, совместимом с примером `techzadanie/sample_submission.csv`.

Используются только открытые библиотеки и модели.

## Требования

- Python 3.10+
- Установленные зависимости из `requirements.txt`:
  - `numpy`
  - `pandas`
  - `sentence-transformers`
  - `faiss-cpu`
  - `tqdm`

## Установка зависимостей

Находясь в корне проекта (где лежат папки `techzadanie` и `solution`), выполните:

```bash
pip install -r solution/requirements.txt
```

## Структура

- `solution/utils.py` — предобработка текста, чанкинг, нормализация эмбеддингов, пути.
- `solution/build_index.py` — построение чанков, эмбеддингов и FAISS-индекса.
- `solution/run_retrieval.py` — загрузка индекса и генерация `submit.csv`.
- `solution/requirements.txt` — зависимости.

Данные берутся из папки `techzadanie` (файлы `websites.csv`, `questions_clean.csv`).

## Шаг 1. Построение индекса

Скрипт `build_index.py`:

- читает `techzadanie/websites.csv`;
- нормализует текст и разбивает его на перекрывающиеся чанки (по ~1000 символов с оверлапом ~200);
- получает эмбеддинги чанков с помощью модели
  `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (только CPU);
- нормализует эмбеддинги и строит FAISS-индекс по косинусному сходству;
- сохраняет:
  - `faiss_index.bin` — бинарный индекс;
  - `chunks_metadata.csv` — соответствие векторов и `web_id`.

Запуск (из корня репозитория или из папки `solution` — пути относительные):

```bash
python solution/build_index.py
```

## Шаг 2. Поиск и формирование submit.csv

Скрипт `run_retrieval.py`:

- загружает `faiss_index.bin` и `chunks_metadata.csv`;
- читает `techzadanie/questions_clean.csv`;
- для каждого вопроса строит эмбеддинг той же моделью;
- ищет top-50 наиболее близких чанков (по индексу);
- агрегирует оценки на уровне документа (`web_id`) и выбирает top-5;
- формирует `submit.csv` в папке `solution` с колонками:
  - `q_id`
  - `web_list` — строка вида `"[935, 687, 963, 1893, 1885]"`.

Запуск:

```bash
python solution/run_retrieval.py
```

## Примечания

- Используются только открытые модели и библиотеки (`sentence-transformers`, `faiss` и др.).
- Генерация ответа (LLM) и этап reranking кросс-энкодером в этом решении опциональны и не реализованы, поскольку метрика оценки — Hit@5 для retriever.


