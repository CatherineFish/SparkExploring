import org.apache.spark.sql.SparkSession

object testStreaming {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: StructuredNetworkWordCount <hostname> <port>")
      System.exit(1)
    }

    val host = args(0)
    val port = args(1).toInt

    val spark = SparkSession
      .builder()
      .appName("testStreaming")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    // Создание DataFrame, представляющего поток входных строк от соединения с host:port
    val lines = spark.readStream
      .format("socket")
      .option("host", host)
      .option("port", port)
      .load()

    // Разбиение строк на слова
    val words = lines.as[String].flatMap(_.split(" "))

    // Генерация текущего количества слов
    val wordCounts = words.groupBy("value").count()

    // Запуск запроса, который выводит текущие счетчики на консоль
    val query = wordCounts.writeStream
      .outputMode("complete")
      .format("console")
      .start()

    query.awaitTermination()
  }
}
