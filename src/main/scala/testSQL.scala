import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.functions.array_contains

object testSQL {

  def AddUpdateSelectExample(spark: SparkSession): Unit = {
    val dataRows = Seq(Row(Row("James;","","Smith"),"36636","M","3000"),
      Row(Row("Michael","Rose",""),"40288","M","4000"),
      Row(Row("Robert","","Williams"),"42114","M","4000"),
      Row(Row("Maria","Anne","Jones"),"39192","F","4000"),
      Row(Row("Jen","Mary","Brown"),"","F","-1")
    )

    val schema = new StructType()
      .add("name",new StructType()
        .add("firstname",StringType)
        .add("middlename",StringType)
        .add("lastname",StringType))
      .add("dob",StringType)
      .add("gender",StringType)
      .add("salary",StringType)

    val df2 = spark.createDataFrame(spark.sparkContext.parallelize(dataRows),schema)
    df2.printSchema()
    df2.show(false)
    df2.select("name").show(false)
    df2.select("name.firstname","name.lastname").show(false)
    df2.select("name.*").show(false)

    //Change the column data type
    df2.withColumn("salary",df2("salary").cast("Integer"))

    //Derive a new column from existing
    val df4=df2.withColumn("CopiedColumn",df2("salary")* -1)

    //Transforming existing column
    val df5 = df2.withColumn("salary",df2("salary")*100)

    //Renaming a column.
    val df3=df2.withColumnRenamed("gender","sex")
    df3.printSchema()

    //Droping a column
    val df6=df4.drop("CopiedColumn")
    println(df6.columns.contains("CopiedColumn"))

    //Adding a literal value
    df2.withColumn("Country", lit("USA")).printSchema()

    //Retrieving
    df2.show(false)
    df2.select("name").show(false)
    df2.select("name.firstname").show(false)
    df2.select("name.*").show(false)

    import spark.implicits._

    val columns = Seq("name","address")
    val data = Seq(("Robert, Smith", "1 Main st, Newark, NJ, 92537"), ("Maria, Garcia","3456 Walnut st, Newark, NJ, 94732"))
    var dfFromData = spark.createDataFrame(data).toDF(columns:_*)
    dfFromData.printSchema()

    val newDF = dfFromData.map(f=>{
      val nameSplit = f.getAs[String](0).split(",")
      val addSplit = f.getAs[String](1).split(",")
      (nameSplit(0),nameSplit(1),addSplit(0),addSplit(1),addSplit(2),addSplit(3))
    })
    val finalDF = newDF.toDF("First Name","Last Name","Address Line1","City","State","zipCode")
    finalDF.printSchema()
    finalDF.show(false)

    df2.createOrReplaceTempView("PERSON")
    spark.sql("SELECT salary*100 as salary, salary*-1 as CopiedColumn, 'USA' as country FROM PERSON").show()
  }

  def FilterExample(spark: SparkSession): Unit = {

    spark.sparkContext.setLogLevel("ERROR")

    val arrayStructureData = Seq(
      Row(Row("James","","Smith"),List("Java","Scala","C++"),"OH","M"),
      Row(Row("Anna","Rose",""),List("Spark","Java","C++"),"NY","F"),
      Row(Row("Julia","","Williams"),List("CSharp","VB"),"OH","F"),
      Row(Row("Maria","Anne","Jones"),List("CSharp","VB"),"NY","M"),
      Row(Row("Jen","Mary","Brown"),List("CSharp","VB"),"NY","M"),
      Row(Row("Mike","Mary","Williams"),List("Python","VB"),"OH","M")
    )

    val arrayStructureSchema = new StructType()
      .add("name",new StructType()
        .add("firstname",StringType)
        .add("middlename",StringType)
        .add("lastname",StringType))
      .add("languages", ArrayType(StringType))
      .add("state", StringType)
      .add("gender", StringType)

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(arrayStructureData),arrayStructureSchema)
    df.printSchema()
    df.show()

    //Condition
    df.filter(df("state") === "OH")
      .show(false)

    //SQL Expression
    df.filter("gender == 'M'")
      .show(false)

    //multiple condition
    df.filter(df("state") === "OH" && df("gender") === "M")
      .show(false)

    //Array condition
    df.filter(array_contains(df("languages"),"Java"))
      .show(false)

    //Struct condition
    df.filter(df("name.lastname") === "Williams")
      .show(false)
  }



  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("testSQL")
      .config("spark.master", "local")
      .getOrCreate()

    //AddUpdateSelectExample(spark)
    FilterExample(spark)
    spark.stop()

  }
}
