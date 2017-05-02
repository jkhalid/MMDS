//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Khalid Jahangeer
 *  @date    Mon Oct  06 23:55:22 EST 2015
 *  @see     LICENSE (MIT style license file).
 */
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import breeze.numerics._
import org.apache.spark.mllib.linalg
object KMeans {
  def main(args: Array[String]) {
    val k = 10   //50
    
    val docfile = "/Users/jkhalid/spark-1.5.0/docword.nips.txt" 

    val final_file = "final.txt"
    val sc = new SparkContext("local","KMeansApp","SPARK_HOME",List("target/scala-2.10/simple-newproject_2.10-1.0.jar"))
    val docData = sc.textFile(docfile,2).filter { x => x.contains(" ") }.cache()
    val newdata = docData.map(x=>{val s = x.split(" "); (s(0),(s(1), s(2)))}  ).groupByKey()
    val finalData = newdata.map({case (k,v) => v.foldLeft(List[String]())({case (acc, (w, c)) => ( {var wRep=List[String](); for(i<-1 to Integer.parseInt(c)) wRep::=w; wRep:::acc })})})
  
       val hashingTF = new HashingTF()
       val tf: RDD[Vector] = hashingTF.transform(finalData)
       tf.cache()
       val idf = new IDF().fit(tf)
       val tfidf = idf.transform(tf)
       .map(v => v.getClass.getMethod("toBreeze").invoke(v).asInstanceOf[breeze.linalg.SparseVector[Double]]).cache()
       tf.unpersist()
       var old_centroids = tfidf.takeSample(withReplacement = false, k)
       val iterations = 100
       var converge = 1.0
       var iter = 0
       var conv = 2.0
       while (iter < iterations && conv > converge)
       { 
         conv = 0.0
         var send_clusters = sc.broadcast(old_centroids)
         var returnval = tfidf.map { x => (close(send_clusters,x),x) } 
         var counter1 = returnval.countByKey
         var sum = returnval.reduceByKey((a,b)=> a+b)
        var new_centre= sum.map({case(k,v)=> {
            var valcount = myFunc(counter1.get(k))
            (k,v:/valcount)
         }
         }).collectAsMap
    var len = new_centre.size
    for((k,v) <- new_centre){
      conv = conv + sqdist(old_centroids(k),v)
      old_centroids(k) = v
    }
    iter = iter + 1
    println()
       }
      println("number of iterations"+iter)
      println("centroid residual left"+conv)
      old_centroids.foreach { println }
}
      
  def myFunc(inVal: Option[Long]): Double = inVal match{
    case Some(x) => x.toDouble
    case None => 0.0
  }
   
  def sqdist(v1:breeze.linalg.SparseVector[Double], v2:breeze.linalg.SparseVector[Double]) :Double ={
    val diff = v1-v2
    diff.dot(diff)
  }
  def close(centre: Broadcast[Array[breeze.linalg.SparseVector[Double]]],p : breeze.linalg.SparseVector[Double]) :Int ={
    var min_dist = Double.MaxValue;
    var min_index = 0;
    var range = centre.value.size
    for (i <- 0 to range-1)
    {
      var distance = sqdist(centre.value(i),p)
      if (distance < min_dist)
      {
        min_dist = distance
        min_index = i
      }
    }
    return min_index
  }
  
}
