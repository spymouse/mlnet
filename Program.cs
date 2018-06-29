using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace mlNet
{

    public class irisData
    {
        // PASSO 1: Defina suas estruturas de dados
        // IrisData é usado para fornecer dados de treinamento
        // input para operações de previsão
        // - As primeiras 4 propriedades são entradas / recursos usados para prever o rótulo
        // - Label é o que você está prevendo e só é definido quando o treinamento
        [Column("0")]
        public float SepalLength;
        [Column("1")]
        public float SepalWidth;
        [Column("2")]
        public float PetalLength;
        [Column("3")]
        public float PetalWidth;
        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }

    // IrisPrediction é o resultado de operações de previsão
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }


    class Program
    {
        static void Main(string[] args)
        {
            // PASSO 2: Crie um pipeline e carregue seus dados
            LearningPipeline pipeline = new LearningPipeline();

            // Se estiver trabalhando no Visual Studio, certifique-se de que o 'Copy to Output Directory'
            // a propriedade do iris-data.txt está definida para 'copiar sempre'
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<irisData>(separator:','));

            // ETAPA 3: Transforme seus dados
             // Atribuir valores numéricos ao texto na coluna "Label", porque somente
             // números podem ser processados durante o treinamento do modelo
            pipeline.Add(new Dictionarizer("Label"));


            // Coloca todos os recursos em um vetor
            pipeline.Add(new ColumnConcatenator("Features","SepalLength", "PetalLength","PetalWidth"));

             // PASSO 4: Adicionar learner
             // Adicione um algoritmo de aprendizado ao pipeline.
             // Este é um cenário de classificação (que tipo de íris é essa?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Converter o rótulo de volta ao texto original (depois de converter em número no passo 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter(){ PredictedLabelColumn = "PredictedLabel"});

            // ETAPA 5: treine seu modelo com base no conjunto de dados
            var model = pipeline.Train<irisData,IrisPrediction>();


             // ETAPA 6: use seu modelo para fazer uma previsão
             // Você pode alterar esses números para testar diferentes previsões
            var prediction = model.Predict(new irisData(){
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });

            Console.WriteLine($"O tipo de flor previsto é: {prediction.PredictedLabels}");

            Console.WriteLine("DotNet Core on Linux :D");
        }
    }
}
