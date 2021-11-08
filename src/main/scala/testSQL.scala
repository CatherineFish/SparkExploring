import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.functions.array_contains

import org.apache.spark.sql.functions._

object testSQL {

  // Пример добавления/обновления/удаления столбцов и выполнения запросов типа select к таблицам
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

    // Изменение типа данных в столбце
    df2.withColumn("salary",df2("salary").cast("Integer"))

    // Создание нового столбца из существующего
    val df4=df2.withColumn("CopiedColumn",df2("salary")* -1)

    // Преобразование существующего столбца
    val df5 = df2.withColumn("salary",df2("salary")*100)

    // Переименование столбца
    val df3=df2.withColumnRenamed("gender","sex")
    df3.printSchema()

    // Удаление столбца
    val df6=df4.drop("CopiedColumn")
    println(df6.columns.contains("CopiedColumn"))

    // Добавление литерального значения
    df2.withColumn("Country", lit("USA")).printSchema()

    // Просмотр столбцов
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

  // Пример выполнения запроса типа Filter к таблицам
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

    // Условие
    df.filter(df("state") === "OH")
      .show(false)

    // SQL выражение
    df.filter("gender == 'M'")
      .show(false)

    // Множественное условие
    df.filter(df("state") === "OH" && df("gender") === "M")
      .show(false)

    // Массив условий
    df.filter(array_contains(df("languages"),"Java"))
      .show(false)

    // Условие на конструкцию
    df.filter(df("name.lastname") === "Williams")
      .show(false)
  }

  // Пример выполнения запроса типа GroupBy к таблицам
  def GroupByExample(spark: SparkSession): Unit = {
    import spark.implicits._

    val simpleData = Seq(("James","Sales","NY",90000,34,10000),
      ("Michael","Sales","NY",86000,56,20000),
      ("Robert","Sales","CA",81000,30,23000),
      ("Maria","Finance","CA",90000,24,23000),
      ("Raman","Finance","CA",99000,40,24000),
      ("Scott","Finance","NY",83000,36,19000),
      ("Jen","Finance","NY",79000,53,15000),
      ("Jeff","Marketing","CA",80000,25,18000),
      ("Kumar","Marketing","NY",91000,50,21000)
    )
    val df = simpleData.toDF("employee_name","department","state","salary","age","bonus")
    df.show()

    df.groupBy("department")
      .agg(
        sum("salary").as("sum_salary"),
        avg("salary").as("avg_salary"),
        sum("bonus").as("sum_bonus"),
        stddev("bonus").as("stddev_bonus"))
      .where(col("sum_bonus") > 50000)
      .show(false)
  }

  // Пример выполнения запроса типа Join к таблицам
  def JoinExample(spark: SparkSession): Unit = {
    spark.sparkContext.setLogLevel("ERROR")

    val emp = Seq((1,"Smith",-1,"2018","10","M",3000),
      (2,"Rose",1,"2010","20","M",4000),
      (3,"Williams",1,"2010","10","M",1000),
      (4,"Jones",2,"2005","10","F",2000),
      (5,"Brown",2,"2010","40","",-1),
      (6,"Brown",2,"2010","50","",-1)
    )
    val empColumns = Seq("emp_id","name","superior_emp_id","year_joined","emp_dept_id","gender","salary")
    import spark.sqlContext.implicits._
    val empDF = emp.toDF(empColumns:_*)
    empDF.show(false)

    val dept = Seq(("Finance",10),
      ("Marketing",20),
      ("Sales",30),
      ("IT",40)
    )

    val deptColumns = Seq("dept_name","dept_id")
    val deptDF = dept.toDF(deptColumns:_*)
    deptDF.show(false)

    empDF.join(deptDF,empDF("emp_dept_id") ===  deptDF("dept_id"),"inner")
      .show(false)
  }


  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("testSQL")
      .config("spark.master", "local")
      .getOrCreate()

    // Запуск примеров. Лучше запускать по одному для более понятного вывода
    AddUpdateSelectExample(spark)
    FilterExample(spark)
    GroupByExample(spark)
    JoinExample(spark)

    spark.stop()
  }
}
