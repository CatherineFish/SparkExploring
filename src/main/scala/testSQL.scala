import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{Row, SparkSession}

object testSQL {

  def SelectExample(spark: SparkSession): Unit = {
    val data = Seq(Row(Row("James","","Smith"),"OH","M"),
      Row(Row("Anna","Rose",""),"NY","F"),
      Row(Row("Julia","","Williams"),"OH","F"),
      Row(Row("Maria","Anne","Jones"),"NY","M"),
      Row(Row("Jen","Mary","Brown"),"NY","M"),
      Row(Row("Mike","Mary","Williams"),"OH","M")
    )

    val schema = new StructType()
      .add("name",new StructType()
        .add("firstname",StringType)
        .add("middlename",StringType)
        .add("lastname",StringType))
      .add("state",StringType)
      .add("gender",StringType)

    val df = spark.createDataFrame(spark.sparkContext.parallelize(data),schema)
    df.printSchema()
    df.show(false)
    df.select("name").show(false)
    df.select("name.firstname","name.lastname").show(false)
    df.select("name.*").show(false)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("testSQL")
      .config("spark.master", "local")
      .getOrCreate()

    SelectExample(spark)
    spark.stop()

  }
}
