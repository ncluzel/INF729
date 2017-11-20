package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegardés précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/


    /** CHARGER LE DATASET **/

    val df: DataFrame = spark
      .read
      .option("header", value = true) // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .parquet("data/prepared_trainingset")


    /** TF-IDF **/

    // On sépare le texte en mots à l'aide d'un RegexTokenizer:
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // On retire les mots qui n'ont pas de sens à l'aide d'un StopWordsRemover:
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // La partie TF est gérée par un CountVectorizer:
    val tf = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    // La partie IDF est gérée par un IDF:
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("tfidf")

    // On réindexe les pays:
    val country_encoder = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // On réindexe les devises:
    val currency_encoder = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    // NB: J'ai remarqué que j'avais quelques fois des bugs d'exécution dues à l'absence de l'étiquette "DE"
    // lors de certains splits training/test. J'utilise donc l'option setHandleInvalid pour éviter qu'elle se reproduise.

    /** VECTOR ASSEMBLER **/

    // On assemble les différentes variables dans une matrice features:
    val vector_assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    /** MODEL **/

    // On définit un modèle de régression logistique:
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/

    // On crée notre pipeline en emboîtant toutes les étapes les unes à la suite des autres:
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, tf, idf,
        country_encoder, currency_encoder, vector_assembler, lr))


    /** TRAINING AND GRID-SEARCH **/

    // On conserve 90% des données pour l'entraînement, et 10% pour le test:
    val Array(training_data, test_data) = df.randomSplit(Array(0.9, 0.1))

    // On crée une grille sur les deux paramètres utilisés pour la validation croisée:
    val paramGrid = new ParamGridBuilder()
      .addGrid(tf.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(1E-8, 1E-6, 1E-4, 1E-2))
      .build()

    // On crée un évaluateur à partir d'un classifieur binaire:
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("final_status")
      .setRawPredictionCol("predictions")

    // On définit les paramètres de notre grille de validation croisée:
    val GridSearchCV = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // On lance la validation croisée sur notre modèle, et on conserve les paramètres offrant la meilleure précision:
    val lrCVModel = GridSearchCV.fit(training_data)

    // On applique la meilleur modèle d'entraînement aux données de test:
    val df_WithPredictions = lrCVModel.transform(test_data)

    // On calcule la précision de notre évaluateur:
    val score = evaluator.evaluate(df_WithPredictions)

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // On sauvegarde le modèle si besoin de le réutiliser à l'avenir:
    lrCVModel.write.overwrite().save("data/lrCVModel")

    println("On remarque que notre classifieur est en quelque sorte 'trop gentil' : " +
      "il a tendance à prédire la réussite de nombreux projets qui ont en réalité échoué.")

    println("On obtient un pourcentage de " + score + "% de prédictions correctes avec notre modèle de régression.")
  }
}