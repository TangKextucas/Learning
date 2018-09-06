import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.ArrayBuffer
/*
import users._  // import everything from the users package
import users.User  // import the class User
import users.{User, UserPreferences}  // Only imports selected members
import users.{UserPreferences => UPrefs}  // import and rename for convenience
*/
object helloworld {
  Logger.getLogger("org").setLevel(Level.ERROR)   //spark日志的输出级别调整
  def main(args: Array[String]): Unit ={
    test_spark_sql()
  }
  //scala基本用法
  def test(): Unit ={
    println("hello "+"world!")    //hello world!
    //var:可变 val:不可变
    println({                     //3
      var x:Int=1+1
      x+1
    })
    //函数：val 函数名=(参数名:参数类型，参数用逗号分隔。也可以不用参数，括号要保留)=>返回值
    val add=(x:Int, y:Int)=>x+y //类似于Java8中的匿名函数
    println(add(2, 4))            //6
    val getTheAnswer=()=>42
    println(getTheAnswer())       //42
    //方法: def 方法名(参数名:参数类型，参数用逗号分隔。可以有多个括号，也可以不用参数，括号不需要保留):返回值类型={
    //      前面可以做更多事情，最后一句是返回值的名字
    //      }
    def a(x: Double,y: Double)(z: Int):String={((x+y)*z).toString}
    println(a(1.2,3.3)(5))        //22.5
    def getSquare(input: Double):String={
      val x=input*input
      x.toString
    }
    println(getSquare(1.732))       //2.999824
    class greeter(prefix:String,suffix:String){
      def greet(name:String):Unit={
        println(prefix+name+suffix)
      }
    }
    val greeter=new greeter("surprise ","!")
    greeter.greet("motherfucker")     //surprise motherfucker!
    val x:List[Int]=List(3, 5, 7)
    val nx = x.map(_*2)
    println(nx)                 //List(6, 10, 14)
    def factorial(x:Int):Int={
      def fac(x:Int,accu:Int):Int={
        if(x<=1) accu
        else fac(x-1,x*accu)
      }
      fac(x,1)
    }
    println("factrial of 10 is "+factorial(10))   //factrial of 10 is 3628800
    def foo(n:Int,v:Int)=for(i<-0 until n;j<-i to n) print(s"($i,$j)")
    foo(5,5)
//    (0,0)(0,1)(0,2)(0,3)(0,4)(0,5)(1,1)(1,2)(1,3)(1,4)(1,5)(2,2)(2,3)(2,4)(2,5)(3,3)(3,4)(3,5)(4,4)(4,5)
  }
  //测试spark:单词计数
  def testSpark(): Unit = {
    val conf=new SparkConf().setMaster("local").setAppName("word count")
    conf.set("spark.testing.memory", "512000000")
    val sc=new SparkContext(conf)
    val input=sc.textFile("C:\\Users\\lenovo\\getright.txt")
//    Can you write a SQL query to output the result for Mary
//    Mary is a student from the college like you
//    naive method:
//    val words=input.flatMap(line=>line.split(" "))
//    val counts=words.map(word=>(word,1)).reduceByKey{case(x,y)=>x+y}
//    counts.foreach(print)
    input.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_+_).collect().foreach(print)
//    (output,1)(is,1)(you,2)(college,1)(a,2)(SQL,1)(to,1)(result,1)(Can,1)(like,1)(query,1)(Mary,2)(for,1)(student,1)(from,1)(write,1)(the,2)
  }
  //测试spark:寻找共同好友
  def findBothFriends():Unit={
    val conf=new SparkConf().setMaster("local").setAppName("findBothFriends")
    conf.set("spark.testing.memory", "512000000")
    val sc=new SparkContext(conf)
    val rddFile=sc.textFile("D:/IntelliJ IDEA/Scala_projects/hello/src/data.log")
//    data.log长这样：
//    解释一下：I把A当做朋友，而A没有把I当做朋友，这里的朋友关系不是相互的
//    而你要找的所有的共同好友就是，这两个人都把那个人当做朋友，那个人就是这两个人的共同好友
//    A:B,C,D,F,E,O
//    B:A,C,E,K
//    C:F,A,D,I
//    D:A,E,F,L
//    E:B,C,D,M,L
//    F:A,B,C,D,E,O,M
//    G:A,C,D,E,F
//    H:A,C,D,E,O
//    I:A,O
//    J:B,O
//    K:A,C,D
//    L:D,E,F
//    M:E,F,G
//    O:A,H,I,J
//读取文本默认按换行符分隔，每一行是一项
    val rddSplit1=rddFile.map(_.split(":"))
//    _是其中一行，每一行按冒号拆分，用map
    rddSplit1.cache()
//    println(rddSplit1.first()(1)+" are "+rddSplit1.first()(0)+"'s friends")
//    B,C,D,F,E,O are A's friends
//    arr是rddSplit1中的每一项,rddSplit1.first()是rddSplit1的第一项
    val rddcom1=rddSplit1.flatMap(arr=>{
      val person=arr(0)
      val friends=arr(1).split(",")
      val paris=new ArrayBuffer[(String,String)]()
      for(i<-0 until friends.length){
        paris+=((friends(i),person))  //被A当做朋友的人 ← A
      // 小括号括起来一个键值对，paris这个变长数组中的每一项是一个(字符串:字符串)的键值对
      }
      paris
      })
//    rddcom1.foreach(print)
//    (B,A)(C,A)(D,A)(F,A)(E,A)(O,A)(A,B)(C,B)(E,B)(K,B)(F,C)(A,C)(D,C)(I,C)(A,D)(E,D)(F,D)(L,D)(B,E)(C,E)(D,E)(M,E)(L,E)(A,F)(B,F)(C,F)(D,F)(E,F)(O,F)(M,F)(A,G)(C,G)(D,G)(E,G)(F,G)(A,H)(C,H)(D,H)(E,H)(O,H)(A,I)(O,I)(B,J)(O,J)(A,K)(C,K)(D,K)(D,L)(E,L)(F,L)(E,M)(F,M)(G,M)(A,O)(H,O)(I,O)(J,O)
      .reduceByKey(_+" "+_)
    //按键累加，键对应的所有值加到一起
//    rddcom1.foreach(println)
//    (L,D E)
//    (G,M)
//    (M,E F)
//    (B,A E F J)
//    (O,A F H I J)
//    (A,B C D F G H I K O)     A ← 所有把A当做朋友的人
//    (I,C O)
//    (K,B)
//    (J,O)       逗号后面大于等于2个人，逗号前面就是他们的共同好友
//    (H,O)       筛选出大于等于2个人的，再把这些人两两组合
//    (C,A B E F G H K)
//    (E,A B D F G H L M)
//    (F,A C D G L M)
//    (D,A C E F G H K L)
//  _1:键, _2:值
      .flatMap(arr=>{
        val persons=arr._2.split(" ")
        val mirage=new ArrayBuffer[String]()
        val friend=arr._1
        if(persons.length>=2){
          for(i<-0 until persons.length-1){     //循环：左闭右开(until)闭(to)
            for(j<-i+1 until persons.length){
              var c=persons(i)+"-"+persons(j)
              mirage+=c
            }
          }
        }
        var fp=new ArrayBuffer[(String,String)]()
        for(j<-mirage.indices){
          fp+=((mirage(j),friend))
        }
        fp
      })
      .reduceByKey(_+" "+_)
      .foreach(println)
    //(B-H,A C E)
    //(F-H,O A C E D)
    //(G-O,A)
    //(F-I,O A)
    //(B-M,E)
    //(C-K,A D)
    //......
  }
  //leetcode第一题
  def leetcode001():Unit={
    def twoSum(nums:Array[Int],target:Int):Array[Int]={
      nums.zipWithIndex.foldLeft(Map.empty[Int,Int])((m,x)=> {
  //foldLeft(一个新的初始值)((新的初始值,调用foldLeft的那个元素)=>一个函数)
  // m:初始为空的哈希表,x:zipWithIndex出来的键值对(数组元素->数组下标)
        if(m.get(target-x._1).isEmpty)
          m+(x._1->x._2)
        else
          return Array(m.getOrElse(target-x._1,-1),x._2)
      })
      null
    }
    val nums=Array(3,2,5,4,7,8,1,0,9,4)
    val target:Int=16
    twoSum(nums,target).foreach(println)
//    nums.zipWithIndex.foreach(print)
    //(3,0)(2,1)(5,2)(4,3)(7,4)(8,5)(1,6)(0,7)(9,8)(4,9)
  }
  def doSomething():Unit={
    val conf=new SparkConf().setMaster("local").setAppName("findBothFriends").set("spark.testing.memory", "512000000")
    val sc=new SparkContext(conf)
    var a=sc.makeRDD(Array(2->"C",3->"b",3->"o",1->"l",3->"y",1->"o",2->"S",1->"v",1->"e"))
    a.reduceByKey(_+_).foreach(println)
//    (1,love)
//    (3,boy)
//    (2,CS)
  }
  def test_spark_sql():Unit={
    val conf = new SparkConf().setMaster("local").setAppName("test_sparkSQL").set("spark.testing.memory", "512000000")
    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()
    import spark.implicits._
    val df = spark.read.json("D:/spark-2.3.1-bin-hadoop2.7/python/test_support/sql/people.json")
    println("read json show------")
    df.show()
    df.printSchema()
    df.select("name").show()
    df.select($"name", $"age"+1).show()
    df.filter($"age">21).show()
    df.groupBy("age").count().show()
    df.createOrReplaceTempView("people")
    println("select * from people")
    val sqlDF = spark.sql("select * from people")
    sqlDF.show()
    println("Register the dataFrame as a global temporary view")
    df.createGlobalTempView("people")
    spark.sql("select * from global_temp.people where age<21").show()
    println("Global temporary view is cross-session")
    spark.newSession().sql("select name from global_temp.people sort by age desc limit 1").show()
  }
}

