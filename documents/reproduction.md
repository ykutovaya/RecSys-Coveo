# Запуск и воспроизведение

Команды выполняются из корня проекта в PowerShell. Проверенная среда — Windows,
Python 3.11; зависимости закреплены в [uv.lock](/uv.lock).
Установка и проверки — в [README](/README.md#установка-и-запуск).

Из чистого клона доступны установка, проверка CSV и подготовка сессий.
Полное воспроизведение обучения и оценки пока требует переноса локальной сборки
свидетельств допуска. Ниже этот разрыв отделён от работающей демонстрации;
готовые данные, модель и каталоги `runs/` в Git не входят.

При локальной проверке `uv sync` взял устаревшую сборку пакета из кеша.
Обычная установка актуального wheel проверена отдельно. Если после изменения
исходников CLI остался старым, из корня проекта пересоберите только этот пакет:

```powershell
uv build --no-build-isolation
uv pip install --python .venv/Scripts/python.exe --no-deps --reinstall `
  dist/coveo_recsys-0.1.0-py3-none-any.whl
```

Зависимости перед этим устанавливаются командой `uv sync` из README. При смене
версии проекта используйте имя нового wheel; удалять весь кеш не требуется.

## Локальная демонстрация

### Необходимые входы

[Настройки сервиса](/configs/coveo_service.json) закрепляют завершённый эксперимент
на 300 кандидатах. Сервису нужны:

| Вход | Как получается |
|---|---|
| `runs/coveo-historical-ag50000-v1/` | `historical build` из подготовленного A_G |
| `runs/coveo-b50000-ranker-v2/` | `train ranker`: модель и `reports/ranker.json` |
| `runs/coveo-b50000-features-v2/reports/features.json` | Сводка `features build`, связанная с моделью; полные таблицы признаков сервису не нужны |
| `data/01_raw/sku_to_content.csv` | Исходная поставка Coveo |
| Профиль поставки, схема признаков и шаблоны объяснений | Файлы репозитория, закреплённые в настройках сервиса |

На компьютере автора эти входы уже готовы. В чистом клоне их сначала нужно
построить; модель для скачивания не опубликована. При собственном расчёте пути
и контрольные суммы берутся из его результатов, а не копируются из старого запуска.
Проверки привязки модели к пакету, признакам и метаданным сохраняются.

### Запуск сервиса

В первом терминале задайте непустой секрет локального отправителя. Его значение
потребуется и во втором терминале; оно не должно попадать в Git или пример ответа.
Для каждого запуска используйте новый `run-dir`.

```powershell
$env:COVEO_LOCAL_TOKEN = Read-Host 'Секрет локального отправителя'
uv run --no-sync coveo-recsys serve `
  --config configs/coveo_service.json `
  --run-dir runs/local-demo-v1 `
  --host 127.0.0.1 --port 8000
```

Текущий календарь сервиса заканчивается **1 января 2027 года, 00:00 UTC**.
Перед более поздней демонстрацией задайте допустимый календарь в отдельной
локальной копии настроек и передайте её в `--config`; не меняйте исторические
периоды обучения. События вне календаря отклоняются. Остановка сервера — `Ctrl+C`.

### Запрос

Во втором терминале выполните следующий пример. Используется действительный
SKU из исторического каталога. Это управляемый просмотр товара на готовой модели,
не воспроизведение покупательской сессии и не новая оценка качества.

```powershell
$env:COVEO_LOCAL_TOKEN = Read-Host 'Тот же секрет локального отправителя'
$uri = 'http://127.0.0.1:8000'
$headers = @{Authorization="Bearer $env:COVEO_LOCAL_TOKEN"; 'X-Stream-Complete'='true'}
Invoke-RestMethod "$uri/health/ready"

$sid = [guid]::NewGuid().ToString('N')
$eventTime = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$event = @{
  source_namespace='local-demo'; session_id=$sid; event_id='e1'
  event_time_ms=$eventTime; event_type='detail'
  sku='0005c742138a6bf913d314c829f41ff2d8e665e9b3670b7b337a9d59b7cff34d'
} | ConvertTo-Json -Compress
$events = '[' + $event + ']'
$sha = [Security.Cryptography.SHA256]::Create()
try {
  $digest = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($events)) |
    ForEach-Object { $_.ToString('x2') })
} finally { $sha.Dispose() }
$head = @{
  source_namespace='local-demo'; session_id=$sid; event_time_ms=$eventTime
  block_id='b1'; event_count=1; event_ids=@('e1'); payload_sha256=$digest; closed=$true
} | ConvertTo-Json -Compress
$body = $head.TrimEnd('}') + ',"events":' + $events + '}'
Invoke-RestMethod "$uri/v1/blocks" -Method Post -Headers $headers `
  -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))

$query = @{
  source_namespace='local-demo'; session_id=$sid
  trigger_block_id='b1'; idempotency_key='q1'
} | ConvertTo-Json -Compress
$response = Invoke-WebRequest "$uri/v1/recommendations" -Method Post -Headers $headers `
  -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($query)) `
  -UseBasicParsing
$answer = [Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray()) |
  ConvertFrom-Json
$answer | Select-Object query_id, items, branch, delivery, reasons | ConvertTo-Json -Depth 5
```

`payload_sha256` вычисляется по точным байтам массива `events`, который отправляется
в теле. Ответ явно читается как UTF-8: это сохраняет русский текст объяснений
в Windows PowerShell 5.1. `X-Stream-Complete` подтверждает завершение передачи блока отправителем;
это не утверждение о полноте исторической телеметрии Coveo.

### Проверенный ответ

11 октября 2026 года выполнен настоящий HTTP-запрос на исходной модели.
Загрузка сервиса — 6,52 с, приём блока — 3,93 мс, получение рекомендаций —
91,62 мс на клиенте. Получены 20 SKU; ниже первые три и выбранные поля ответа:

```json
{
  "items": [
    "0005c742138a6bf913d314c829f41ff2d8e665e9b3670b7b337a9d59b7cff34d",
    "52cf0c63ee45fba1d2d76bf59639f48fcc06233c58b9fd0251d3a3013acc4a1b",
    "0e2c9bc9cdd664a3d5169e29464b450d960d02a05900705438a812a5fbd19ef1"
  ],
  "branch": "primary",
  "delivery": "delivered",
  "reasons": ["search:search_absent"]
}
```

HTTP-ответ совпал с записью в `requests.jsonl` каталога запуска. Полный журнал
содержит внутреннюю диагностику и не публикуется. Повтор просмотренного товара
допустим: исключаются купленные, а не все просмотренные SKU. Названия товаров
неизвестны. Время одного запроса не является P99 или гарантией под нагрузкой.

## Порядок полного расчёта

### Данные, A_G и B

Получите Public Data Release 1.0.0 по [инструкции Coveo](https://github.com/coveooss/SIGIR-ecom-data-challenge#license),
приняв условия самостоятельно. В `data/01_raw/` нужны `browsing_train.csv`,
`search_train.csv`, `sku_to_content.csv`; архив переносить не требуется.
Схема и ожидаемые контрольные суммы — в [профиле](/configs/coveo_supply_v1.json).

Ниже команды новой подготовки; они читают большие CSV и **не нужны для повторного
запуска готового API**. При повторении используйте новые имена каталогов.

```powershell
uv run --no-sync coveo-recsys data inspect-supply `
  --input-dir data/01_raw --profile configs/coveo_supply_v1.json `
  --run-id supply-new --run-dir runs/supply-new
uv run --no-sync coveo-recsys data prepare `
  --input-dir data/01_raw --profile configs/coveo_supply_v1.json `
  --plan configs/coveo_preparation.json `
  --run-id ag50000-new --run-dir runs/ag50000-new
```

Историческая сборка должна ссылаться на **новый** результат, а не на прежний A_G
из опубликованных настроек. Создание локальной копии с фактическим хешем:

```powershell
New-Item -ItemType Directory -Path runs/reproduction-settings -ErrorAction Stop
$utf8 = New-Object System.Text.UTF8Encoding($false)
$historical = Get-Content configs/coveo_historical.json -Raw -Encoding UTF8 | ConvertFrom-Json
$historical.preparation.path = 'runs/ag50000-new'
$historical.preparation.report_sha256 = (Get-FileHash `
  runs/ag50000-new/reports/preparation.json -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
  (Join-Path (Get-Location) 'runs/reproduction-settings/historical.json'),
  ($historical | ConvertTo-Json -Depth 30), $utf8)
uv run --no-sync coveo-recsys historical build `
  --config runs/reproduction-settings/historical.json `
  --run-id historical-new --run-dir runs/historical-new

$plan = Get-Content configs/coveo_preparation.json -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($period in $plan.data.periods) {
  $period.sample_size = if ($period.name -eq 'B') { 50000 } else { 0 }
}
[IO.File]::WriteAllText(
  (Join-Path (Get-Location) 'runs/reproduction-settings/b-plan.json'),
  ($plan | ConvertTo-Json -Depth 30), $utf8)
uv run --no-sync coveo-recsys data prepare `
  --input-dir data/01_raw --profile configs/coveo_supply_v1.json `
  --plan runs/reproduction-settings/b-plan.json `
  --run-id b50000-new --run-dir runs/b50000-new
```

Периоды, seed, правила допуска и пределы ресурсов сохраняются. Отбор не зависит
от наличия покупки; недобор допустим только при недостатке допущенных сессий.
Подготовка B не открывает доступ к его признакам и обучению.

### Обучение и оценка: пока не автономная инструкция

Следующая таблица показывает существующие команды и их входы, **не заменяет
отсутствующую сборку допусков**. Форматы настроек — в [configs/README.md](/configs/README.md).

| Шаг | Команда | Необходимые входы |
|---|---|---|
| Допуск B | `manifest validate --stage preflight` | Паспорт, фактические проверки B, результаты тестов и привязки к коду, плану, схемам и рецепту |
| Признаки B | `features build --period B` | Принятый B, исторический пакет, схема 93 признаков, настройки отбора и действительный допуск |
| Модель | `train ranker` | Те же входы, готовые признаки B, рецепт LightGBM |
| Проверка C_smoke | `data prepare --evaluation-config ...`, затем `replay --period C_smoke` | Закреплённый план, готовая модель, допуск `post_training`; после подготовки — её фактический хеш |
| Без метаданных | `sensitivity metadata --step features`, затем `--step train` | Тот же B, отдельные схема, рецепт, квоты, допуск и каталоги результата |
| D | `data prepare --evaluation-config ...`, затем `evaluate` | Обе модели, заранее закреплённый план `pre_D`, затем подготовленный D; после оценки — `post_D` |

Каждая вычислительная команда принимает собственные `--config`, `--run-dir`
и `--run-id`. Выбор через C_select не обязателен для воспроизведения исходной
оценки D: вариант 370 исследован позднее и не заменяет основную модель.

Открытое препятствие: свидетельства проверок и паспорта реальных запусков
собирались локальными вспомогательными скриптами, которые не входят в репозиторий.
Часть этих скриптов зависит от прежнего B на 10 000 сессий и истории запусков.
Команда `manifest init` создаёт шаблон, а `manifest validate` проверяет свидетельства,
но ни одна не создаёт их вместо реальных проверок.

Для автономного воспроизведения нужно перенести прямую проверку нового B и сборку
свидетельств из актуальных результатов тестов без зависимости от старых запусков.
Копирование прежних паспортов, подстановка фиктивных хешей или отключение допуска
не решают задачу. Пока эта часть не упакована, обещать полный запуск из чистого
клона нельзя. В текущей проверке публикации полный расчёт не повторялся.

## Измеренные затраты

Измерения завершённых локальных запусков на Windows, не прогноз для другого
компьютера. Загрузка, проверки и запись включены в стоимость команды, если
не сказано иное. Каталоги ниже — источники локальных результатов, не ссылки
на скачиваемые артефакты.

| Шаг | Время | Пиковая память процесса | Источник в `runs/` |
|---|---:|---:|---|
| Возобновление сборки пакета A_G, 50 000 сессий | 370 с | 493 МиБ | `coveo-historical-ag50000-v1/` |
| Подготовка B, 50 000 сессий | 34 мин 52 с | 1 725 МиБ | `coveo-b50000-v1/` |
| Признаки B основной модели | 1 ч 49 мин 17 с | 1,12 ГиБ | `coveo-b50000-features-v2/` |
| Обучение основной модели с проверками | 35 мин 20 с | 2,00 ГиБ | `coveo-b50000-ranker-v2/` |
| Оценка C_smoke, 50 000 сессий | 2 ч 31 мин | 3,81 ГиБ | `coveo-csmoke50000-replay-v2/` |
| Подготовка D, 60 000 сессий | 21 мин 35 с | 344 МиБ | `coveo-d60000-v2/` |
| Оценка D, четыре метода | 3 ч 12 мин 16 с | 837 МиБ | `coveo-d60000-evaluate-v2/` |
| Сравнение 300/370 на C_select | 3 ч 38 мин 5 с | 706 МиБ | `candidate370-v1/comparison/` |

Замер A_G относится к возобновлению, а не к полной сборке из CSV с нуля.
Признаки основной модели занимают 1,52 ГиБ, модель со сводкой — 0,72 МиБ,
две таблицы оценки D — 1,71 ГиБ. Для подготовки 50 000 сессий установлены пределы
4 часа, 8 ГиБ памяти, 4 ГиБ временного диска; это не обещание фактического расхода.

Источники качества: `runs/coveo-d60000-evaluate-v2/reports/replay.json`,
`runs/coveo-d-plan-v2/manifest-post_D.json`,
`runs/candidate370-v1/comparison/reports/replay.json`.
Итоги без сырых событий приведены в [README](/README.md#измеренные-результаты).
Повтор оценки уже рассмотренного D воспроизводит прежний эксперимент,
но не даёт нового независимого подтверждения улучшений.
