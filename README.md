# SparkExploring
Несколько примеров, демонстрирующих возможности Spark.

Рассмотрены некоторые возможности трёх модулей Spark: Spark MLib, Spark SQL, Spark Streaming.

Работа с каждым модулем вынесена в отдельный объект, а конкретные примеры - функции в объекте: 
```bash
├── build.sbt
├── data
│   └── libsvm_data.txt
├── project
├── src
│   ├── main
│   │   └── scala
│   │       ├── testMLib.scala
│   │       ├── testSQL.scala
│   │       └── testStreaming.scala
│   └── test
└── target
```

Примеры запускаются из `def main`, их удобнее запускать по одному, чтобы понимать, что выводится.

## Подключение Spark 
В файл `build.sbt` добавлено:

```bash
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0",
)
```

## Spark MLib 
1. CorrelationExample - подсчёт корелляционной матрицы с использованием коэффициента корелляции Пирсона и Спирмена.
2. ChiSquareTestExample - проверка гипотез: Хи-квадрат Пирсона.
3. PipelineExample - создание ML Pipeline.
4. RandomForestClassifierExample - логистическая регрессия: RandomForest.

## Spark SQL
1. AddUpdateSelectExample - добавление/обновление/удаление столбцов и выполнение select.
2. FilterExample - выполнение filter.
3. GroupByExample - выполнение groupby.
4. JoinExample - выполнение join.

## Spark Streaming
В примере происходит подсчёт количества слов в текстовых данных, полученных от сервера, прослушивающего сокет TCP. 
Чтобы запустить данный пример необходимо:
1. Запустить в отдельном терминале
``` bash
$ nc -lk 9999
```
2. Запустить объект testStreaming с аргументами 
``` bash
localhost 9999
```
> В IntelliJ IDEA необходимо добавить аргументы в конфигурацию запуска: Run > Edit Configurations > Program arguments
