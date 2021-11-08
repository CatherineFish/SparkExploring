import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

import org.apache.spark.ml.stat.ChiSquareTest

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.sql.SparkSession


object testMLib {
  // Пример подсчёта корреляционной матрицы
  def CorrelationExample(spark: SparkSession): Unit = {
      import spark.implicits._

      val data = Seq(
        Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
        Vectors.dense(4.0, 5.0, 0.0, 3.0),
        Vectors.dense(6.0, 7.0, 0.0, 8.0),
        Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
      )

      val df = data.map(Tuple1.apply).toDF("features")
      // Коэффициент корелляции Пирсона
      val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
      println(s"Pearson correlation matrix:\n $coeff1")
      // Коэффициент корелляции Спирмена
      val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
      println(s"Spearman correlation matrix:\n $coeff2")

    }

    // Пример проверки гипотез - Хи-квадрат Пиросна
    def ChiSquareTestExample(spark: SparkSession): Unit = {
      import spark.implicits._

      val data = Seq(
        (0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))
      )

      val df = data.toDF("label", "features")
      val chi = ChiSquareTest.test(df, "features", "label").head
      println(s"pValues = ${chi.getAs[Vector](0)}")
      println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
      println(s"statistics ${chi.getAs[Vector](2)}")

  }

  /*
  Пример конвейера машинного обучения:
     * Разделить текст каждого документа (просто строка в примере) на слова
     * Преобразовать слова в числовой вектор признаков
     * Обучить модель (логистическая регрессия)
  */
  def PipelineExample(spark: SparkSession): Unit = {
    // Подготовка train, представляющих собой кортежи (id, text, label)
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    // Конфигурация ML Pipeline: токенизатор, hashingTF, логистическая регрессия
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Применение fit к train
    val model = pipeline.fit(training)

    // Можно сохранить текущее состояние pipeline
    model.write.overwrite().save("/tmp/spark-logistic-regression-model")

    // До применения fit тоже можно было сохранить pipeline
    pipeline.write.overwrite().save("/tmp/unfit-lr-model")

    // Можно загрузить сохраненное состояние pipeline
    val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

    // Подготовка test - кортежи без меток (id, text)
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    // Предсказание на test
    model.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

  }

  // Пример логистической регрессии - RandomForest
  def RandomForestClassifierExample(spark: SparkSession): Unit = {
    // загрузка и парсинг данных, преобразование в DataFrame
    val data = spark.read.format("libsvm").load("data/libsvm_data.txt")

    // Индексирование меток путём добавления метаданных в столбец меток
    // Обучение на всём датасете
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Автоматическое индексирование категориальных признаков, их индексация
    // Установка значения maxCategories
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Разделение данных на train (30%) и test
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Обучение модели RandomForest
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Конвертирование индексированных меток обратно в исходные
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))

    // Формирование Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Обучение модели
    val model = pipeline.fit(trainingData)

    // Предсказание
    val predictions = model.transform(testData)

    // Выбор строк для демонстрации
    predictions.select("predictedLabel", "label", "features").show(5)

    // Выбор (prediction, true label) и вычисление ошибки на test
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
  }


  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("testMLib")
      .config("spark.master", "local")
      .getOrCreate()

    // Запуск примеров. Лучше запускать по одному для более понятного вывода
    CorrelationExample(spark)
    ChiSquareTestExample(spark)
    PipelineExample(spark)
    RandomForestClassifierExample(spark)

    spark.stop()

  }
}